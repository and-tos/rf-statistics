from pathlib import Path
import subprocess as sp


def resubmit(lsf_out_file):
    filename = lsf_out_file.name.split("_")
    del filename[0]
    del filename[-1]

    filename = "lsf_" + "_".join(filename) + ".sh"
    if filename == "lsf_contacts.sh":
        bsub_file = lsf_out_file.parent.parent / "lsf.sh"
    else:
        bsub_file = lsf_out_file.parent.parent / filename
    with open(bsub_file, "r") as bf:
        bsub_str = bf.read()
    with open(bsub_file, "w") as bf:
        bf.write(bsub_str)
    filter_out = (
        sp.check_output(
            "bsub",
            stdin=open(bsub_file, "r"),
        )
        .rstrip()
        .decode("utf-8")
        .split("<")[1]
        .split(">")[0]
    )
    print(filename)
    print(filter_out)
    lsf_out_file.unlink()
    return


def main():
    for lsf_out_file in Path("rf_statistics").glob("*/lsf_output/lsf_*.out"):
        with open(lsf_out_file, "r") as lsf:
            lsf_string = lsf.read()
            for error_string in [
                "KeyboardInterrupt",
                "FileNotFoundError",
                "RuntimeError",
            ]:
                if error_string in lsf_string:
                    resubmit(lsf_out_file)
                    break


if __name__ == "__main__":
    main()
