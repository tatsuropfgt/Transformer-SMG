import argparse
import os

import muspy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    args = parser.parse_args()

    # dirs = os.listdir(args.dir)
    # for file in dirs:
    #     if file.endswith(".mid"):
    #         music = muspy.read_midi(os.path.join(args.dir, file))
    #         muspy.write_audio(
    #             os.path.join(args.dir, file.replace(".mid", ".wav")), music
            # )

    for root, dirs, files in os.walk(args.dir):
        for file in files:
            if file.endswith(".mid"):
                music = muspy.read_midi(os.path.join(root, file))
                muspy.write_audio(
                    os.path.join(root, file.replace(".mid", ".wav")), music
                )


if __name__ == "__main__":
    main()
