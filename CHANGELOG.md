# Changelog

Todos los cambios notables de este proyecto se documentan en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/),
y este proyecto se adhiere a [Semantic Versioning](https://semver.org/lang/es/).

## [Unreleased]

Sin cambios pendientes de publicar.

## [1.0.0] - 2026-08-31

Primera versión estable, según `pyproject.toml`. Incluye el flujo de
documento único original y la incorporación posterior del motor de dossier
headless.

### Added

* **Flujo de documento único** (`src/graph/workflow.py`,
  `src/graph/nodes/single_document.py`): extracción, validación
  determinista y reintento dirigido vía recorte de ROI y un VLM, con
  escalado a revisión humana (`interrupt()` de LangGraph) cuando se agota el
  presupuesto de reintentos.
  * Dominio (`src/domain/`): modelos Pydantic (`IDCardSchema`,
    `InvoiceSchema`, `GenericFormSchema`, `PageSchema`,
    `ProofOfAddressSchema`), estado del flujo (`state.py`), políticas
    (`policy.py`), herramientas de validación determinista
    (`validation_tools.py`) y preprocesamiento de imagen con OpenCV/PIL
    (transposición EXIF, varianza de Laplaciano con abandono temprano,
    unsharp masking, CLAHE en espacio LAB, redimensionado y recorte de ROI).
  * Puertos (`src/ports/`): `DocumentExtractorPort`, `VLMProviderPort` y
    `PDFProcessorPort` como contratos ABC.
* **Motor de dossier headless** (`src/graph/orchestrator.py`,
  `src/graph/worker_graph.py`): procesamiento de PDFs multi-documento en
  paralelo mediante un fan-out Map-Reduce de LangGraph (`Send`), sin
  suspensión en ningún punto del flujo.
  * `smart_clustering`: rasteriza cada página a 300 DPI, la clasifica sobre
    una copia reducida y agrupa páginas no contiguas en documentos lógicos
    (por ejemplo, un INE fotocopiado en páginas separadas por otro
    documento).
  * `router`, `doc_intel`, `vlm_grounding`, `validation`, `vlm_retry`,
    `unresolved_handler` y `consolidation`: subgrafo por documento con
    reintento de campos vía recorte y VLM, y terminal headless
    `REQUIRES_MANUAL_ENTRY` cuando el presupuesto de reintentos se agota.
  * Contrato de salida JSON del dossier (`dossier_status`,
    `documents[].unresolved_fields[].bounding_box`, etc.) en
    `src/domain/orchestrator_state.py`.
  * Recorte de ROI dirigido con reescalado bicúbico 2.5x, denoising (Fast
    Non-Local Means), ecualización CLAHE y unsharp masking, aplicado solo al
    recorte del reintento del worker de dossier.
* Adaptadores de extracción para Azure AI Document Intelligence, AWS
  Textract, Google Document AI y un extractor VLM nativo
  (`src/infrastructure/adapters/extractors/`).
* Adaptadores VLM para OpenAI, Anthropic, Ollama y Microsoft AI Foundry
  (`src/infrastructure/adapters/vlm/`), incluyendo autenticación secretless
  vía `DefaultAzureCredential` en los adaptadores de Azure.
* Adaptador de procesamiento de PDF `PyMuPDFProcessor` (rasterización a 300
  DPI) en `src/infrastructure/adapters/processors/`.
* Raíz de composición (`src/infrastructure/config.py`,
  `build_dependencies()`) que selecciona los adaptadores concretos a partir
  de las variables de entorno `EXTRACTOR_PROVIDER` y `VLM_PROVIDER`.
* CLI (`main.py`) con los modos `--image` / `--resume` para el flujo de
  documento único y `--dossier` / `--id` para el motor de dossier.
* Checkpointing persistente en `main.py` mediante `SqliteSaver`
  (`checkpoints.sqlite3`), permitiendo reanudar una ejecución de documento
  único suspendida en una invocación de CLI posterior.
* Inyección del almacén de certificados del sistema operativo vía
  `truststore.inject_into_ssl()` en `main.py`, para evitar fallos de
  verificación TLS detrás de proxies corporativos que interceptan HTTPS.
* Suite de pruebas (`tests/`) con 265 pruebas que cubren la cadena de
  preprocesamiento, las herramientas de validación, la lógica de mapeo de
  adaptadores, las reglas de arquitectura y ambos grafos de extremo a
  extremo, sin credenciales ni acceso a red gracias a fakes inyectados a
  través de los ports.
* `tests/test_architecture.py`: analiza el código fuente para hacer cumplir
  la regla de dependencias hacia adentro y prohibir `interrupt()` en
  cualquier módulo del motor de dossier.

### Changed

* Reestructuración de `src/graph/nodes.py` en el paquete `src/graph/nodes/`,
  separando cada nodo del grafo de documento único
  (`src/graph/nodes/single_document.py`) y del worker de dossier en módulos
  independientes, más utilidades compartidas (`_shared.py`).
* `src/domain/policy.py` (`WorkflowPolicy`) extendido con la tabla de
  enrutamiento por tipo de documento y los parámetros del motor de dossier.
* `src/infrastructure/config.py` ampliado para construir las dependencias de
  ambos flujos (documento único y dossier) desde una única raíz de
  composición.
* Los adaptadores de extracción priorizan el texto crudo impreso sobre el
  valor normalizado del proveedor, para no interferir con los validadores
  basados en expresiones regulares.
