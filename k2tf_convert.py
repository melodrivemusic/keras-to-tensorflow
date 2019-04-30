import argparse

from k2tf import convertGraph

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", help="REQUIRED: The HDF5 Keras model you wish to convert to .pb")
    parser.add_argument("--outdir", "-o", default="./output", help="The directory to place the output files - default('./output')")
    parser.add_argument("--prefix", "-p", default="out", help="The prefix for the output aliasing - default('out')")
    parser.add_argument("--name", "-n", default="model.pb", help="The name of the resulting output graph - default('model.pb')")
    args = parser.parse_args()

    convertGraph(args.model, args.outdir, args.prefix, args.name)
