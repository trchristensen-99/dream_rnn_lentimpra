# Setting Up This Repository on GitHub

This guide will help you push this repository to your GitHub account.

## Step 1: Create a New Repository on GitHub

1. Go to https://github.com/new
2. Repository name: `dream_rnn_lentimpra` (or your preferred name)
3. Description: "DREAM-RNN training for lentiMPRA data"
4. Choose Public or Private (your preference)
5. **Do NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

## Step 2: Initialize and Push

From the `dream_rnn_lentimpra` directory:

```bash
cd dream_rnn_lentimpra

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: DREAM-RNN training for lentiMPRA"

# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/dream_rnn_lentimpra.git

# Push to GitHub
git push -u origin main
# Or if your default branch is 'master':
git push -u origin master
```

## Step 3: Verify

1. Go to your repository on GitHub: `https://github.com/YOUR_USERNAME/dream_rnn_lentimpra`
2. Verify that:
   - `train_DREAM_RNN_lentiMPRA.py` is present
   - `DREAM_RNN_lentiMPRA.yaml` is present
   - `README.md` is present
   - `TRAINING_GUIDE.md` is present
   - Data files are **NOT** included (they're in .gitignore)
   - Results are **NOT** included

## Step 4: Share with Collaborators (Optional)

1. Go to repository Settings → Collaborators
2. Add users who need access
3. They can then clone:
   ```bash
   git clone https://github.com/YOUR_USERNAME/dream_rnn_lentimpra.git
   ```

## Important Notes

- Data files are excluded via `.gitignore` (users download from Zenodo)
- Training results are excluded (users generate their own)
- Only code, configs, and documentation are tracked

## Updating the Repository

After making changes:

```bash
git add .
git commit -m "Description of changes"
git push origin main  # or master
```

## Repository Structure

```
dream_rnn_lentimpra/
├── train_DREAM_RNN_lentiMPRA.py    # Main training script
├── DREAM_RNN_lentiMPRA.yaml        # Training configuration
├── README.md                        # Quick start guide
├── TRAINING_GUIDE.md               # Complete training guide
├── SETUP_GITHUB.md                 # This file
├── requirements.txt                 # Python dependencies
├── .gitignore                      # Excludes data and results
└── LICENSE                         # MIT License (if included)
```
