import numpy as np
import pandas as pd
import json


def read_json(file):
    with open(file, 'r') as string:
        return json.load(string)


def save_track(track):
    ep = pd.DataFrame(track, columns=["position_x", "position_y", "shift_x", "shift_y"])
    ep.to_pickle("./tracks/track_" + str(ep["position_x"].shape[0]) + ".pkl")


def get_shifts(tiles):
    shift_tiles = dict()
    for key in tiles.keys():
        xs = tiles[key][0]
        ys = tiles[key][1]
        shift_tiles[key] = []
        shift_tiles[key].append([(j - i) for i, j in zip(xs, xs[1:])])
        shift_tiles[key].append([(j - i) for i, j in zip(ys, ys[1:])])

    return shift_tiles


def create_track_from_points(tiles, kinds):
    track_x = [0.5]
    track_y = [0.5]

    for kind in kinds:
        list_shift_x = tiles[kind][0]
        list_shift_y = tiles[kind][1]
        for idx, x in enumerate(list_shift_x):
            track_x.append(track_x[idx] - list_shift_x[idx])
        for idx, y in enumerate(list_shift_y):
            track_y.append(track_y[idx] - list_shift_y[idx])

    shift_x = [(j - i) for i, j in zip(track_x, track_x[1:])]
    shift_y = [(j - i) for i, j in zip(track_y, track_y[1:])]

    track = list(zip(track_x, track_y, shift_x, shift_y))

    return track


tiles = read_json('tiles.json')
shift_tiles = get_shifts(tiles)
track = create_track_from_points(shift_tiles, ['curve_right'])
save_track(track)




