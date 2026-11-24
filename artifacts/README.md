# Artefactos exportados

| Archivo | Formato | Tamaño | Ida y vuelta (máx. |Δp| frente a PyTorch) |
|---|---|---|---|
| `nagfuse_seed00.onnx` | ONNX, opset 17 | 360 MB | 9.2 × 10⁻⁶ |
| `nagfuse_seed00.torchscript.pt` | TorchScript | 361 MB | 1.8 × 10⁻⁷ |
| `nagfuse_seed00_meta.json` | metadatos (clases, normalización, s_lo, ρ, π0, opset, versiones) | < 1 KB | — |

Los pesos superan el límite de 100 MB por archivo de GitHub, por eso **no están en el repositorio**: se guardan
en el almacenamiento externo del proyecto (Google Drive, carpeta `E3_TrashNet/export/`) y se declaran aquí, como
permite la Sección 10.1 del enunciado. Copie aquí `nagfuse_seed00_meta.json` (pequeño, sí se versiona).

**Prueba de ida y vuelta.** Se exportó la semilla 0 de NAGFuse, se ejecutó el mismo lote de imágenes en PyTorch,
TorchScript y ONNX Runtime (CPU) y se midió la mayor diferencia absoluta en las probabilidades de salida. Ambas
quedan por debajo del umbral de 10⁻⁴ del enunciado. Para exportar a ONNX se reescribieron dos operaciones sin
cambiar su resultado: la mediana del estimador MAD (por ordenamiento) y `logaddexp` de la mezcla (forma estabilizada
con el máximo).

**Uso en el dashboard.** Copie `nagfuse_seed00.onnx` a `dashboard/modelos/` o defina `NAGFUSE_ONNX=<ruta>`.
Solo se exportó NAGFuse; los demás modelos se muestran en el dashboard con resultados precalculados (Sección 9.2).
