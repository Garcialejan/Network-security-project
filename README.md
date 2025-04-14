## Data Validation Workflow

 El objetivo principal es garantizar que los datos ingresados cumplan con las expectativas definidas en el esquema (schema) y detectar cualquier desviación (data drift) antes de entrenar modelos.

```mermaid
graph TD
    A[Data Validation Config] --> B[Ingested]
    B --> C{Schema → 10 features}
    C --> D[Read Data]
    D --> E{Validate number of Columns}
    E --> F{Is numerical columns exist}
    F --> G[Train Status]
    F --> H[Test Status]
    G --> I{Status}
    H --> J{Status}
    I --> K[Validation Status]
    J --> K
    K --> L[Detect Dataset Drift]
    L --> M[Distribution]
    K --> N[Validation Error]

    subgraph Data Validation Artifact
        O[Artifacts]
        P[Time Stamp]
        Q[Data Validation]
        R[report.json]
    end

    %% Anotaciones adicionales
    style C fill:#ffcccc,stroke:#f66,stroke-width:2px
    style L fill:#ccffcc,stroke:#6f6,stroke-width:2px
    style N fill:#ffcccc,stroke:#f66,stroke-width:2px