
import json

from visualize import SkeletonPlayer

if __name__ == "__main__":
    frames = []
    with open(f"code/daa/demo2.json", "r") as f:
        data = json.load(f)
    player = SkeletonPlayer(
        data["cameras"], data["frames"], data["fps"], data["center"], data["up"]
    )
    player.show()
