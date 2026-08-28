LabelForge v0.0.1

LabelForge prototype with labeling and reproducible training workspaces.

Enthalten:
- Label Workspace
- Training Workspace with isolated Facemap/DeepLabCut environments
- local and JUSUF/Slurm training bundles
- Help-Button
- Create Base Model
- Refine Existing Model
- Specialize Base Model
- validated manifests, environment files and launch scripts

Training Workspace:
1. Check or install the isolated backend environment.
2. Select parent model, training data and labels/config.
3. Configure the new model version and training parameters.
4. Enter the JUSUF Slurm account/resources.
5. Generate one immutable training bundle.

The JUSUF integration uses SSH/SFTP plus Slurm. It does not store passwords,
browser sessions or Jupyter tokens. Facemap can bundle a versioned project
training adapter; DeepLabCut bundles a direct config-driven runner.

Start:
1. Öffne Anaconda Prompt.
2. Aktiviere dein Python-Environment.
3. Installiere PySide6:
   pip install PySide6
4. Wechsle in diesen Ordner.
5. Starte:
   python app.py
