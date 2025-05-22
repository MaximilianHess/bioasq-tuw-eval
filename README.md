# BioASQ TUW Evaluation

## Steps to Run the Evaluation

### 1. Clone the Repository

### 2. Install Pixi (Dependency Manager)

**Linux / macOS:**

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

**Windows (PowerShell):**

```powershell
iwr -useb https://pixi.sh/install.ps1 | iex
```

### 3. Configure the Evaluation

Edit the `config.yaml` file:

* **`predictions_path`**: Path to your prediction files (must match the BioASQ challenge format).
* **`batch`**: Batch number to evaluate (default is `4`, the latest released batch).
* **`result_name`** *(optional)*: Custom name for the results; otherwise, a timestamp will be used.

### 4. Run the Evaluation Script

```bash
pixi run eval
```
When executing this command for the first time, it will automatically install the required dependencies. 

### 5. View Results

Results will be printed in the console and saved in the `results/` directory.

### IMPORTANT NOTE
This evaluation was currently only tested on **macOS with ARM architecture**. If you are using a different OS or architecture, please report any issues you encounter and feel free to open a pull request.