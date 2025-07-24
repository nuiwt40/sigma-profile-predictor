import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import RDLogger
import matplotlib.pyplot as plt
from PIL import Image
import io

# --- Optional import for 3D visualization ---
try:
    import py3Dmol
    from st_py3dmol import st_py3dmol

    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False
# ----------------------------------------------------

# --- Optional import for Ketcher drawing ---
try:
    from streamlit_ketcher import st_ketcher

    KETCHER_AVAILABLE = True
except ImportError:
    KETCHER_AVAILABLE = False
# ----------------------------------------------------


# --- CONFIGURATION ---
# This must match the configuration used during training
FP_RADIUS = 2
FP_NBITS = 2048
# Define the sigma bins (x-axis) for both plotting and downloading
SIGMA_BINS = np.arange(-0.025, 0.0251, 0.001)

# --- UPDATED: Use the new multi-output model filename ---
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = os.path.join(script_dir, 'best_xgboost_multi_output_model.pkl')
# ---------------------------------------------------------

# --- SETUP: Suppress RDKit warnings ---
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


# --------------------------------------

# --- Helper Functions ---
def generate_fingerprint(smiles, radius, n_bits):
    """
    Generates a Morgan fingerprint for a given SMILES string.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except Exception:
        return None


def generate_3d_structure(smiles):
    """
    Generates a 3D structure from a SMILES string and returns it as a PDB block.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=1)
        AllChem.MMFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)
        pdb_block = Chem.MolToPDBBlock(mol)
        return pdb_block
    except Exception:
        return None


# Function to load the trained model
@st.cache_resource
def load_model():
    """
    Loads the trained model from the .pkl file.
    """
    if not os.path.exists(MODEL_FILENAME):
        st.error(f"Fatal Error: Model file not found at '{MODEL_FILENAME}'")
        st.info("Please ensure 'best_multi_output_model.pkl' is in the same folder as this script.")
        st.stop()
    try:
        with open(MODEL_FILENAME, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


# Function to format the output .txt file
def format_profile_for_download(sigma_bins, profile_values):
    """
    Combines the sigma bins with the predicted values and formats them
    for download.
    """
    output_string = ""
    for bin_val, prof_val in zip(sigma_bins, profile_values):
        output_string += f" {bin_val: .15E}  {prof_val: .15E}\n"
    return output_string


# --- Streamlit App UI ---

st.set_page_config(page_title="Property Predictor", layout="wide")
st.title("🧪 Sigma Profile Predictor")
st.write("Draw a molecule or enter a SMILES string to predict its properties using the trained multi-output model.")

model = load_model()

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("Input Molecule")

    # --- Integrated Ketcher Drawing Widget ---
    # Use session state to sync the drawing and text input
    if 'smiles' not in st.session_state:
        st.session_state.smiles = "CCO"  # Initial default

    if KETCHER_AVAILABLE:
        st.write("Draw a molecule:")
        # The Ketcher component returns the SMILES string of the drawn molecule
        drawn_smiles = st_ketcher(st.session_state.smiles)
        # If the drawing changes, update the session state
        if drawn_smiles != st.session_state.smiles:
            st.session_state.smiles = drawn_smiles
            st.rerun()  # Rerun to update the text input below
    else:
        st.info("To enable drawing, please install streamlit-ketcher (`pip install streamlit-ketcher`).")

    # The text input's value is now controlled by the session state
    smiles_input = st.text_input("SMILES string:", key="smiles")
    # ----------------------------------------------------

    if st.button("Predict Properties"):
        if smiles_input:
            fingerprint = generate_fingerprint(smiles_input, FP_RADIUS, FP_NBITS)

            if fingerprint is not None:
                # The model now only needs the fingerprint
                combined_prediction = model.predict(fingerprint.reshape(1, -1))[0]

                # --- Unpack Area, Volume, and Profile ---
                predicted_area = combined_prediction[0]
                predicted_volume = combined_prediction[1]
                predicted_profile = combined_prediction[2:]

                # Store results in session state to persist them
                st.session_state.area = predicted_area
                st.session_state.volume = predicted_volume
                st.session_state.profile = predicted_profile
                st.session_state.smiles_for_prediction = smiles_input
                st.success("Prediction successful!")
            else:
                st.error("Invalid SMILES string. Please check your input.")
                st.session_state.area = None
        else:
            st.warning("Please enter a SMILES string.")
            st.session_state.area = None

    # Display 3D or 2D molecule image if valid
    st.subheader("Molecule Structure")
    if smiles_input:
        if PY3DMOL_AVAILABLE:
            pdb_block = generate_3d_structure(smiles_input)
            if pdb_block:
                view = py3Dmol.view(width=400, height=300)
                view.addModel(pdb_block, 'pdb')
                view.setStyle({'stick': {}})
                view.setBackgroundColor('0xeeeeee')
                view.zoomTo()
                st_py3dmol(view)
            else:
                st.warning("Could not generate 3D structure. Displaying 2D instead.")
                mol = Chem.MolFromSmiles(smiles_input)
                if mol:
                    img = Draw.MolToImage(mol, size=(300, 300))
                    st.image(img, caption="2D Structure")
        else:
            mol = Chem.MolFromSmiles(smiles_input)
            if mol:
                img = Draw.MolToImage(mol, size=(300, 300))
                st.image(img, caption="2D Structure")

with col2:
    st.subheader("Predicted Properties")
    if 'area' in st.session_state and st.session_state.area is not None:
        predicted_area = st.session_state.area
        predicted_volume = st.session_state.volume
        predicted_profile = st.session_state.profile
        smiles_for_title = st.session_state.smiles_for_prediction

        # --- UPDATED: Display Area and Volume metrics ---
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric(label="Predicted Area (Å²)", value=f"{predicted_area:.2f}")
        metric_col2.metric(label="Predicted Volume (Å³)", value=f"{predicted_volume:.2f}")

        # Plot the sigma profile
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(SIGMA_BINS, predicted_profile, 'o-', label=f'Predicted Profile', color='green')
        ax.set_title(f'Predicted Sigma Profile for {smiles_for_title}')
        ax.set_xlabel('Sigma (e/Å²)')
        ax.set_ylabel('p(σ) Value')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)

        # --- Download Button ---
        st.write("---")
        st.subheader("Export Sigma Profile")
        download_data = format_profile_for_download(SIGMA_BINS, predicted_profile)

        # --- Filename includes Area and Volume ---
        st.download_button(
            label="📥 Download Profile as .txt",
            data=download_data,
            file_name=f"predicted_profile_Area_{predicted_area:.2f}_Vol_{predicted_volume:.2f}_{smiles_for_title}.txt",
            mime="text/plain"
        )
    else:
        st.info("Draw or enter a molecule and click 'Predict' to see the results here.")
