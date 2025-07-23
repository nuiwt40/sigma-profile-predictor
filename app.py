import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import matplotlib.pyplot as plt
from PIL import Image
import io
import xgboost as xgb

# --- NEW: Optional import for RDKit Drawing ---
# This prevents the app from crashing if the drawing module fails to import
try:
    from rdkit.Chem import Draw
    RDKIT_DRAW_AVAILABLE = True
except ImportError:
    RDKIT_DRAW_AVAILABLE = False
# ----------------------------------------------------

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
FP_RADIUS = 2
FP_NBITS = 2048
SIGMA_BINS = np.arange(-0.025, 0.0251, 0.001)

script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = os.path.join(script_dir, 'xgboost_model.pkl')
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

@st.cache_resource
def load_model():
    """
    Loads the trained model from the .pkl file.
    """
    if not os.path.exists(MODEL_FILENAME):
        st.error(f"Fatal Error: Model file not found at '{MODEL_FILENAME}'")
        st.info("Please ensure 'xgboost_model.pkl' is in the same folder as this script.")
        st.stop()
    try:
        with open(MODEL_FILENAME, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

def format_profile_for_download(sigma_bins, profile_values):
    """
    Formats the predicted profile for download.
    """
    output_string = ""
    for bin_val, prof_val in zip(sigma_bins, profile_values):
        output_string += f" {bin_val: .15E}  {prof_val: .15E}\n"
    return output_string

# --- Streamlit App UI ---

st.set_page_config(page_title="XGBoost Property Predictor", layout="wide")
st.title("🧪 XGBoost Property Predictor")
st.write("Draw a molecule or enter a SMILES string, provide a temperature, and predict its properties.")

model = load_model()

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("Input Parameters")

    if 'smiles' not in st.session_state:
        st.session_state.smiles = "CCO"

    if KETCHER_AVAILABLE:
        st.write("Draw a molecule:")
        drawn_smiles = st_ketcher(st.session_state.smiles)
        if drawn_smiles != st.session_state.smiles:
            st.session_state.smiles = drawn_smiles
            st.rerun()
    else:
        st.info("To enable drawing, please install streamlit-ketcher (`pip install streamlit-ketcher`).")

    smiles_input = st.text_input("SMILES string:", key="smiles")
    temp_input = st.number_input("Temperature (K):", value=298.15, format="%.2f")

    if st.button("Predict Properties"):
        if smiles_input and temp_input:
            fingerprint = generate_fingerprint(smiles_input, FP_RADIUS, FP_NBITS)
            
            if fingerprint is not None:
                combined_input = np.concatenate((fingerprint, [temp_input])).reshape(1, -1)
                combined_prediction = model.predict(combined_input)[0]
                
                st.session_state.vcosmo = combined_prediction[0]
                st.session_state.profile = combined_prediction[1:]
                st.session_state.smiles_for_prediction = smiles_input
                st.success("Prediction successful!")
            else:
                st.error("Invalid SMILES string. Please check your input.")
                st.session_state.vcosmo = None
        else:
            st.warning("Please enter a SMILES string and temperature.")
            st.session_state.vcosmo = None

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
            # Fallback to 2D if 3D generation fails
            elif RDKIT_DRAW_AVAILABLE:
                st.warning("Could not generate 3D structure. Displaying 2D instead.")
                mol = Chem.MolFromSmiles(smiles_input)
                if mol:
                    img = Draw.MolToImage(mol, size=(300, 300))
                    st.image(img, caption="2D Structure")
        # Fallback to 2D if 3D library is not available
        elif RDKIT_DRAW_AVAILABLE:
            st.info("To view molecules in 3D, please install py3Dmol and streamlit-py3dmol.")
            mol = Chem.MolFromSmiles(smiles_input)
            if mol:
                img = Draw.MolToImage(mol, size=(300, 300))
                st.image(img, caption="2D Structure")
        else:
            st.warning("Could not display molecule structure. RDKit drawing module failed to import.")


with col2:
    st.subheader("Predicted Properties")
    if 'vcosmo' in st.session_state and st.session_state.vcosmo is not None:
        predicted_vcosmo = st.session_state.vcosmo
        predicted_profile = st.session_state.profile
        smiles_for_title = st.session_state.smiles_for_prediction
        
        cleaned_profile = np.maximum(predicted_profile, 0)
        
        st.metric(label="Predicted Vcosmo (Å³)", value=f"{predicted_vcosmo:.2f}")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(SIGMA_BINS, cleaned_profile, 'o-', label=f'Predicted Profile', color='purple')
        ax.set_title(f'Predicted Sigma Profile for {smiles_for_title}')
        ax.set_xlabel('Sigma (e/Å²)')
        ax.set_ylabel('p(σ) Value')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)
        
        st.write("---")
        st.subheader("Export Sigma Profile")
        download_data = format_profile_for_download(SIGMA_BINS, cleaned_profile)
        st.download_button(
            label="📥 Download Profile as .txt",
            data=download_data,
            file_name=f"predicted_profile_Vcosmo_{predicted_vcosmo:.2f}_{smiles_for_title}.txt",
            mime="text/plain"
        )
    else:
        st.info("Draw or enter a molecule, provide a temperature, then click 'Predict' to see the results.")
