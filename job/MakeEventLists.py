from pathlib import Path


def MakeList(mode, pressure, diff):
    # Define the local directory path
    local_path = f"/ospool/ap40/data/krishan.mistry/job/ATPC/Pressure/{mode}/{pressure}/{diff}"
    osdf_prefix = "osdf://"

    # Get all files in the directory
    file_list = [f"{file.name}" for file in Path(local_path).iterdir() if file.is_file()]

    # Write to a text file
    output_file = f"filelists/{mode}_{pressure}_{diff}.txt"
    with open(output_file, "w") as f:
        f.write("\n".join(file_list))

    print(f"File list saved to {output_file}")


modes = ["ATPC_0nubb", "ATPC_Bi", "ATPC_Tl"]
pressures = ["1bar", "5bar", "10bar", "15bar"]
diffs = ["0.05percent", "0.1percent", "0.25percent", "0.5percent", "5percent", "nodiff"]

for m in modes:
    for p in pressures:
        
        if (p != "1bar"):
            diffs = ["5percent", "nodiff"]
        else:
            diffs = ["0.1percent", "0.25percent", "5percent", "nodiff"]
        
        for d in diffs:
            MakeList(m, p, d)


