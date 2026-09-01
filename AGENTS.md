# AGENTS.md

Guía para agentes de codificación (Junie, Codex, Claude, etc.) que trabajen en
este repositorio. Léela antes de tocar código: describe la arquitectura, las
reglas que el propio test suite hace cumplir, y los comandos que debes usar
para validar cualquier cambio.

## 1. Qué es este proyecto

ImageExtractor es un sistema de OCR agnóstico de proveedor construido con:

* **LangGraph** para la orquestación de los flujos.
* **Arquitectura Hexagonal (Ports & Adapters)** para el desacoplamiento.
* **OpenCV / PIL** para el preprocesamiento de imágenes.
* **Pydantic** para los contratos de datos.

Expone dos flujos desde una única raíz de composición (`build_dependencies()`
en `src/infrastructure/config.py`):

* **Documento único** (`src/graph/workflow.py`): extrae, valida y, si un campo
  falla, recorta esa región y pide a un VLM que la vuelva a leer. Si se agota
  el presupuesto de reintentos, escala a un humano vía `interrupt()` de
  LangGraph.
* **Dossier** (`src/graph/orchestrator.py`): toma un PDF multi-documento, lo
  divide en documentos lógicos y los procesa **en paralelo** (Map-Reduce). Es
  estrictamente **headless**: nunca se suspende; un documento no validable se
  reporta como `REQUIRES_MANUAL_ENTRY`.

Consulta `README.md` para el detalle completo de ambos grafos, el diagrama de
directorios, el contrato de salida del dossier y las notas de producción.

## 2. Regla de dependencias (obligatoria)

El código está organizado en capas que apuntan **hacia adentro**, y
`tests/test_architecture.py` lo hace cumplir parseando el código fuente. Antes
de añadir un import, verifica que respeta esto:

* `src/domain/` y `src/ports/` no importan ningún SDK de proveedor, ningún
  módulo de `src/infrastructure/`, ni siquiera `langgraph`.
* `src/graph/` importa solo `domain` y `ports`. Los nodos reciben un
  `WorkflowPolicy` (un dataclass de dominio plano), nunca el `Settings`
  respaldado por variables de entorno.
* `src/infrastructure/config.py` es la única raíz de composición que decide
  qué adaptador concreto se inyecta. `compile_workflow()` accede a ella solo
  mediante un import local a la función, que es la única costura de
  composición permitida.
* Ningún módulo del motor de dossier (`src/graph/orchestrator.py`,
  `src/graph/worker_graph.py`, los nodos del worker) puede llamar a
  `interrupt()`: el dossier es headless por contrato, no solo por convención.

Si tu cambio necesita romper una de estas reglas, es casi seguro que el código
está en la capa equivocada: muévelo en vez de relajar la regla.

## 3. Layout del código

```text
src/
├── domain/            # Núcleo: modelos Pydantic, estado, políticas, validación, preprocesamiento
├── ports/              # Contratos (ABC): DocumentExtractorPort, VLMProviderPort, PDFProcessorPort
├── infrastructure/     # Adaptadores concretos (Azure, AWS, Google, OpenAI, Anthropic, Ollama, PyMuPDF) + config.py
└── graph/              # Nodos y grafos de LangGraph (workflow.py, orchestrator.py, worker_graph.py, edges.py)
tests/                  # Suite de pytest, incluyendo test_architecture.py
main.py                 # CLI: modo documento único (con --resume) y modo --dossier
```

## 4. Configuración del entorno

```bash
pip install -r requirements.txt
cp .env.example .env
```

Los adaptadores importan su SDK de forma perezosa, así que solo necesitas
instalar los extras del proveedor que vayas a usar (`openai`, `anthropic`,
`azure`, `azure-foundry`, `aws`, `google` en `pyproject.toml`).

Dos variables de entorno deciden toda la infraestructura:

```bash
EXTRACTOR_PROVIDER=azure        # azure | aws | google | native_vlm
VLM_PROVIDER=azure_foundry      # openai | anthropic | ollama | azure_foundry
```

El resto de las variables (umbrales de preprocesamiento, presupuesto de
reintentos, rutas de estrategia por tipo de documento, credenciales por
proveedor) están documentadas con comentarios en `.env.example`. No dupliques
esos valores por defecto en código nuevo: léelos desde `Settings`
(`src/infrastructure/config.py`) y pásalos a través de `WorkflowPolicy`.

## 5. Ejecutar la suite de pruebas

```bash
python -m pytest
```

* Ningún test requiere credenciales ni acceso a red: la suite inyecta fakes a
  través de los ports (`tests/conftest.py`).
* Antes de dar por cerrado cualquier cambio en `domain/`, `ports/`, `graph/` o
  `infrastructure/`, ejecuta al menos:
  * `python -m pytest tests/test_architecture.py` — para confirmar que no
    rompiste la regla de dependencias.
  * El módulo de test específico del área tocada (por ejemplo
    `tests/test_workflow.py`, `tests/test_dossier_workflow.py`,
    `tests/test_validation_tools.py`, `tests/test_adapters.py`,
    `tests/test_clustering.py`, `tests/test_router.py`,
    `tests/test_unresolved_handler.py`, `tests/test_preprocessing.py`,
    `tests/test_config.py`).
* Si añades un caso de uso nuevo (tipo de documento, proveedor, rama de
  routing), añade o extiende un test en el módulo correspondiente; no dejes
  comportamiento nuevo sin cobertura.

## 6. Estilo de código

* Sigue `ruff` con la configuración de `pyproject.toml`
  (`line-length = 100`, `target-version = "py311"`). Ejecuta `ruff check .`
  antes de considerar terminado un cambio si la herramienta está disponible.
* Usa `from __future__ import annotations` y type hints completos, como hace
  el resto del código (`main.py`, los nodos de `graph/`, los modelos de
  `domain/models/`).
* Los docstrings son estilo Google/reST cortos, en inglés, explicando el
  "por qué" cuando la decisión no es obvia (mira los docstrings de
  `src/domain/preprocessing.py` o `main.py` como referencia). Mantén ese
  idioma y ese nivel de detalle en el código nuevo.
* Los errores de proveedor se envuelven en las excepciones de dominio propias
  del port (`ExtractionError`, `VLMProviderError`, `PDFProcessingError`); no
  dejes escapar excepciones nativas del SDK fuera del adaptador.

## 7. Añadir un nuevo tipo de documento

Todo vive en la capa de dominio; no se tocan adaptadores ni nodos:

1. Añade el esquema Pydantic en `src/domain/models/`, declarando `ALIASES`
   para el vocabulario de cada proveedor.
2. Regístralo en `SCHEMA_REGISTRY` (`src/domain/models/__init__.py`).
3. Registra sus reglas de campo y reglas de negocio en
   `src/domain/validation_tools.py`.
4. Mapea el tipo de documento a un id de modelo de proveedor (Azure
   `prebuilt-*`, un processor id de Google) vía configuración.
5. Si el motor de dossier también debe reconocerlo, añádelo a `PageType` /
   `LogicalDocType` en `src/domain/orchestrator_state.py` y dale una rama en
   `DEFAULT_STRATEGY_BY_DOC_TYPE` (o sobrescríbelo vía `STRATEGY_BY_DOC_TYPE`).

Añade siempre un test que cubra la ruta feliz y al menos un caso de fallo de
validación para el tipo nuevo.

## 8. Añadir un nuevo proveedor (extractor o VLM)

1. Implementa el ABC correspondiente (`DocumentExtractorPort` o
   `VLMProviderPort`) en `src/infrastructure/adapters/`.
2. Importa el SDK del proveedor de forma perezosa (dentro del constructor o
   del método, no al nivel de módulo), siguiendo el patrón de los adaptadores
   existentes.
3. Registra el proveedor en la fábrica de `src/infrastructure/config.py`
   (`build_dependencies()`), leyendo su configuración desde `Settings`.
4. Añade sus variables de entorno a `.env.example`, con comentarios
   explicando valores por defecto y modo secretless si aplica (ver el bloque
   de Azure como referencia).
5. Añade un fake del proveedor en `tests/conftest.py` y tests en
   `tests/test_adapters.py`.

## 9. Actualizar CHANGELOG.md

Cuando tu cambio sea observable por un usuario del CLI o de la librería
(`main.py`, `src/graph/workflow.py`, `src/graph/orchestrator.py`, el contrato
de salida del dossier, nuevas variables de entorno), añade una entrada bajo
`[Unreleased]` en `CHANGELOG.md`, siguiendo el formato de
[Keep a Changelog](https://keepachangelog.com/). No reescribas entradas ya
publicadas de versiones anteriores.

## 10. Qué NO hacer

* No importes `langgraph`, un SDK de proveedor, ni módulos de
  `infrastructure/` desde `domain/` o `ports/`.
* No llames a `interrupt()` desde ningún módulo del motor de dossier.
* No pases `Settings` (el objeto respaldado por variables de entorno)
  directamente a un nodo del grafo; pasa siempre `WorkflowPolicy`.
* No escribas datos de documentos de identidad a disco: todo el pipeline es
  en memoria por diseño (ver la nota de PII en `README.md`, sección 9).
* No debilites ni saltes tests fallidos (`skip`, `xfail`, comentar
  aserciones) para hacer pasar la suite; corrige la causa real.
