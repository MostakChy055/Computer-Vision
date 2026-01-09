## Parameter Fine-tuning
<img width="849" height="380" alt="image" src="https://github.com/user-attachments/assets/25cb7288-5fda-47ce-8b5d-a59e3b2c5066" />
<img width="1181" height="713" alt="image" src="https://github.com/user-attachments/assets/1269dbdf-e2d1-40fb-82ac-0d050f1be1f0" />

### Observation
- Here reconstruction works but the issue is how the model treats the words.
  - The model treats both the text strokes and noise as high frequency objects so moves to remove both of them.
  - This leads to clean background
  - Characters collapse into blobs / barcode-like patterns
- Used MSE (.5 weightage) as loss function. What this does is tries to find mean, which results in oversmooth result.
  - MSE assumes best output is mean of all results
  - This discourages sharp edges
  - This encourages smoothness
  - This destroys thin structure
- Another fundamental issue here is core architecure: **Under-complete AE**
  - Compress stroke topology
  - Lose precise spatial relationship
  - Prefers textures over shape
    - These result in: character losing indentity and model re-constructing letter like structures
- Even decreasing the weightage of MSE to **0.3** didn't help, rather it increased the loss!
<img width="941" height="406" alt="image" src="https://github.com/user-attachments/assets/f0053a16-a026-40d9-8210-7b670f1f964e" />
<img width="1215" height="734" alt="image" src="https://github.com/user-attachments/assets/ddcfaf5e-485b-4757-a996-22204257b623" />


