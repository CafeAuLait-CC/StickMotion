import os
import sys

workspace_path = os.path.abspath(os.path.join(__file__, *[".."] * 3))
print(f"Workspace path: {workspace_path}")
os.chdir(workspace_path)
sys.path.insert(0, workspace_path)
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
from torch import cuda
from scipy.ndimage import gaussian_filter1d
from inter_model import get_pose_model

device = "cuda" if cuda.is_available() else "cpu"
generate_pose = get_pose_model(
    cfg_path="configs/remodiffuse/remodiffuse_kit.py",
    weight_path="stickman/weight/kit_ml/last.ckpt",
    device=device,
)
app = FastAPI()
templates = Jinja2Templates(directory="stickman/interaction")

# Buffer for storing tracks saved from the web interface
track_cache = {}


def resample_polyline(polyline, num_samples):
    # Equidistant sampling
    # polyline [n, 2]
    # num_samples int
    distances = np.sqrt(np.sum(np.diff(polyline, axis=0) ** 2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    total_distance = cumulative_distances[-1]

    sample_distances = np.linspace(0, total_distance, num_samples)

    # Calculate the coordinates of the sampling points using interpolation.
    sample_points = np.zeros((num_samples, 2))
    sample_points[:, 0] = np.interp(
        sample_distances, cumulative_distances, polyline[:, 0]
    )
    sample_points[:, 1] = np.interp(
        sample_distances, cumulative_distances, polyline[:, 1]
    )

    return sample_points


def smooth_track(noised_track):
    x, y = noised_track[:, 0], noised_track[:, 1]
    sigma = 4
    x_smooth = gaussian_filter1d(x, sigma=sigma, mode="nearest")
    y_smooth = gaussian_filter1d(y, sigma=sigma, mode="nearest")
    smoothed_track = np.vstack((x_smooth, y_smooth)).T
    return smoothed_track


def get_interpolated_track(ori_tracks, point_num=100):
    tracks = []
    for track in ori_tracks:  # [_points, 2] to [point_num, 2]
        track = np.array(track)
        # track = smooth_track(track)
        interpolated_track = resample_polyline(track, point_num)
        tracks.append(interpolated_track)

    tracks = np.array(tracks)
    return tracks


def norm_tracks(tracks):
    assert isinstance(tracks, np.ndarray)
    a = tracks.copy()
    tracks = (a - a.mean(axis=(0, 1))) / (
        a.max(axis=(0, 1)) - a.min(axis=(0, 1))
    ).mean()
    return tracks


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/submit")
async def submit(request: Request):
    data = await request.json()
    stickmen = []
    for stickname in ["stickman"]:
        lines = []
        for _lines in data[stickname]:
            line = []
            for point in _lines:
                line.append([point["x"], -point["y"]])
            lines.append(line)
        if len(lines) == 0:
            stickmen.append([])
            continue
        elif len(lines) == 6:
            lines = get_interpolated_track(lines, point_num=64)
            lines = norm_tracks(lines)
            stickmen.append(lines)
        else:
            raise ValueError("The number of the lines should be 6 or 0")
    # user_input = (stickmen, text)
    buf = generate_pose(stickmen[0])  # (6,64,2)
    return StreamingResponse(buf, media_type="image/png")


@app.post("/save_track")
async def save_track(request: Request):
    """Cache the drawn stickman as start/middle/end track."""
    data = await request.json()
    stage = data.get("stage")
    lines_data = data.get("stickman", [])
    lines = []
    for _lines in lines_data:
        line = []
        for point in _lines:
            line.append([point["x"], -point["y"]])
        lines.append(line)
    if len(lines) != 6:
        return JSONResponse(
            {"error": "The number of the lines should be 6"}, status_code=400
        )

    track = get_interpolated_track(lines, point_num=64)
    track = norm_tracks(track)
    track_cache[stage] = track

    return JSONResponse({"status": "cached", "stage": stage})


@app.post("/save_tracks")
async def save_tracks():
    """Concatenate cached tracks and save to an npy file."""
    zero_track = np.zeros((6, 64, 2), dtype=np.float32)
    ordered = [
        track_cache.get("start", zero_track),
        track_cache.get("middle", zero_track),
        track_cache.get("end", zero_track),
    ]
    save_arr = np.stack(ordered, axis=0)
    save_path = os.path.abspath("stickman_tracks.npy")
    np.save(save_path, save_arr)
    track_cache.clear()
    return JSONResponse({"status": "saved", "path": save_path})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=12004)
