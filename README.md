Download the FLIR dataset from the following link:

<https://adas-dataset-v2.flirconservator.com/#downloadguide>

Extract the contents of the zipped folder into ./FLIR_ADAS_v2

-----

Install miniconda with:


```
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o miniconda.exe
start /wait "" miniconda.exe /S
del miniconda.exe

```

Open Anaconda Powershell Prompt (miniconda3) and run:


```
conda init powershell
```

Now, open powershell from the main folder in your local copy of this repository. Run:

```
conda create --prefix ./envs/[name_of_environment]
```

After that finishes there will be some text output that tells you the command to run to activate the environment. Whenever you want to activate this environment to run code, use that command. It's something like:

```
conda activate C:\...\...\...\...\...\ECE_523_Final_Project\envs\ [name_of_environment]
```

You can also select it as your Python environment in vscode when running Python code.