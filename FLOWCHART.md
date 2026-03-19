# LOL-Blur Dataloader Flowchart

## Initialization Flow

```mermaid
flowchart TD
    A[Start: __init__ called] --> B[Set input_dir = root/low_blur_noise]
    B --> C[Set gt_dir = root/high_sharp_original]
    C --> D{input_dir exists?}
    D -->|No| E[Raise RuntimeError]
    D -->|Yes| F{gt_dir exists?}
    F -->|No| E
    F -->|Yes| G[Initialize empty lists:<br/>input_paths, gt_paths]
    G --> H[Loop: For each subfolder in input_dir]
    H --> I{Is subfolder a directory?}
    I -->|No| H
    I -->|Yes| J[Loop: For each file in subfolder]
    J --> K{File is image?<br/>.png/.jpg/.jpeg}
    K -->|No| J
    K -->|Yes| L[Check if matching GT file exists]
    L --> M{GT file exists?}
    M -->|No| N[Print Warning: Skip file]
    M -->|Yes| O[Add input_path to input_paths]
    O --> P[Add gt_path to gt_paths]
    N --> J
    P --> Q{More files?}
    Q -->|Yes| J
    Q -->|No| R{More subfolders?}
    R -->|Yes| H
    R -->|No| S{Any pairs found?}
    S -->|No| T[Raise RuntimeError]
    S -->|Yes| U[Define Transform Pipeline:<br/>Resize → ToTensor → Normalize]
    U --> V[Print Dataset Statistics]
    V --> W[End: Dataset Ready]
```

## Data Retrieval Flow (__getitem__)

```mermaid
flowchart TD
    A[Start: __getitem__ called with idx] --> B[Get input_path from input_paths[idx]]
    B --> C[Get gt_path from gt_paths[idx]]
    C --> D[Open input image: PIL Image.open]
    D --> E[Convert input to RGB]
    E --> F[Open GT image: PIL Image.open]
    F --> G[Convert GT to RGB]
    G --> H[Apply transform to input:<br/>Resize → ToTensor → Normalize]
    H --> I[Apply transform to GT:<br/>Resize → ToTensor → Normalize]
    I --> J[Extract filename from input_path]
    J --> K[Return Dictionary:<br/>input, gt, filename]
    K --> L[End]
```

## Main Execution Flow

```mermaid
flowchart TD
    A[Start: Main block] --> B[Set DATASET_PATH]
    B --> C[Create LOLBlurDataset instance]
    C --> D[Dataset __init__ executes]
    D --> E[Create DataLoader:<br/>batch_size=4, shuffle=True]
    E --> F[Initialize batch counter: i=0]
    F --> G[Loop: For each batch in DataLoader]
    G --> H{i >= 2?}
    H -->|Yes| M[Break loop]
    H -->|No| I[Get batch from DataLoader]
    I --> J[Print batch info:<br/>- Input shape<br/>- GT shape<br/>- Filenames]
    J --> K[Increment i]
    K --> G
    M --> N[Print Success Message]
    N --> O[End]
```

## Complete System Flow

```mermaid
flowchart LR
    A[Dataset Root] --> B[low_blur_noise/]
    A --> C[high_sharp_original/]
    B --> D[Subfolder 1]
    B --> E[Subfolder 2]
    C --> F[Subfolder 1]
    C --> G[Subfolder 2]
    D --> H[img1.png]
    D --> I[img2.png]
    E --> J[img3.png]
    F --> K[img1.png]
    F --> L[img2.png]
    G --> M[img3.png]
    
    H -.->|Match| K
    I -.->|Match| L
    J -.->|Match| M
    
    K --> N[Dataset Pairs]
    L --> N
    M --> N
    
    N --> O[DataLoader]
    O --> P[Batches]
    P --> Q[Model Training]
```
