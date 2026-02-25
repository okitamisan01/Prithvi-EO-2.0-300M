## Step1: Try Inference

```
python inference.py \
  --config_path config.json \
  --checkpoint Prithvi_EO_V2_300M.pt \
  --rgb_outputs
```

- output: in `output'

## Step2: Attention Pooling and get Q

- code: eo_extract_features.py (base: inference.py)
- code: eo_attention_pooling.py


### eo_extract_features.py

**Base Script:** `inference.py` (from Prithvi-EO-2.0)
**Purpose:** Extraction of Earth Observation (EO) Embedding ($Q$) for County-Level Crop Yield Prediction.

#### Overview
The original `inference.py` was intended for the task of reconstructing missing parts of images as a Masked Autoencoder (MAE).
In this script, the inference pipeline is modified and redefined as a process not for image reconstruction but for **extracting and saving the "high-dimensional features (Latent Representation) acquired by the Prithvi-EO encoder"**.

#### Processing Pipeline

1. **Spatial Patching via Sliding Window**
   Receives a huge satellite image (GeoTIFF) as input and divides it into non-overlapping patches (windows) of `224x224` pixels that the model can process. This covers vast geographical boundaries such as a County.

   **Note: Temporal Context**
   Each `224x224` patch is processed while retaining data from 4 different time steps (e.g., Spring, Summer, Autumn, Winter).

3. **Feature Extraction from Final Layer**
   In the `run_model` function, instead of performing image restoration by the decoder, the `forward_features()` method of the frozen Prithvi-EO encoder is called.
   The **"output of the final layer"** immediately after passing through all 24 blocks of the Transformer is obtained as the most abstracted semantic information (Earth Observation Embedding) of the target patch.

4. **Saving Query ($Q$) (Saving Tensors)**
   The extracted features are saved as individual `.pt` files (e.g., `extracted_q_patch_0.pt`) for each patch to be used as **Query ($Q$)** in the subsequent Cross-Modal Multi-Head Attention.
   * **Output Shape:** `[1, 785, 1024]`
     * `1`: Batch Size
     * `785`: Tokens (1 CLS Token + 784 Spatio-temporal Tokens)
     * `1024`: Embedding Dimension


### eo_attention_pooling.py

**Implementation:** Custom Attention Pooling Module
**Purpose:** Integrates variable-length Earth Observation (EO) embeddings divided into multiple patches to generate a County Embedding ($Q$), which is a representative vector at the county level.

#### Processing Pipeline

1. **Loading Varying-length Sequence**
   Batch loads the features for each patch extracted by `eo_extract_features.py` (e.g., `extracted_q_patch_0.pt` ~ `8.pt`).

2. **Patch Representation**
   Extracts the leading CLS token from each patch tensor (`[1, 785, 1024]`) and treats it as a group of vectors `[1, 1024]` representing the spatio-temporal meaning of that patch.

3. **Attention Weighting**
   Evaluates the importance of each patch using a lightweight attention network based on Multi-Layer Perceptron (MLP).

4. **Permutation-Invariant Aggregation**
   Performs a weighted sum independent of the patch order to condense the variable-length sequence into a single fixed-length tensor.
   * **Output Shape:** `[1, 1024]`
   * This output becomes the **Query ($Q$)** for fusion with weather data in the subsequent Cross-Modal Multi-Head Attention.