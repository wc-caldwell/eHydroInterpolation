import subprocess
from pathlib import Path

# Get all standard element files and put them in order
se_list = Path("notebooks").glob("SE*")

# Run each standard element in order
### ------------------------------------------------------ ###
###          If you have other types of files              ###
### you will need to add code and dependencies to run them ###
### ------------------------------------------------------ ###

for se_path in sorted(se_list):
    print(f"Running {se_path}")
    try:
        if se_path.suffix == ".ipynb":
            subprocess.run(
                ["jupyter", "nbconvert", "--execute",
                 "--to", "notebook", "--inplace", str(se_path)],
                check=True, capture_output=True, text=True
            )
            
        elif se_path.suffix == ".Rmd":
            subprocess.run(
                ["Rscript", "-e", f'library(knitr);knit("{se_path}")'],
                check=True, capture_output=True, text=True
            )
            
        else:
            print("File type not supported...skipping.")
            
    except subprocess.CalledProcessError as e:
        print(f"Error running {se_path}:")
        print(e.stdout)
        print(e.stderr)
        raise

print("Workflow complete")
