# Model card: NAGFuse

## Resumen
NAGFuse (*Noise-Adaptive Gated Fusion*) clasifica fotos de residuos en 6 clases (cardboard, glass, metal, paper, plastic, trash). Mezcla las probabilidades de ViT-B/16 y EfficientNet-B0 con un peso que fija un estimador analítico del nivel de ruido de la imagen.

| Campo | Valor |
|---|---|
| Autores | Anyely Sanjuan Benavides y Juan David Díaz Fernández (Universidad del Norte, 2026) |
| Versión | E3, tag `e3-protocolo-v3`; artefacto exportado: semilla 0 |
| Arquitectura | ViT-B/16 + EfficientNet-B0 (ImageNet-1k, torchvision), estimador MAD sobre Haar, contracción de ondícula en la rama CNN, compuerta $g(\hat\sigma)$ sin parámetros entrenables |
| Parámetros | 89 818 505 |
| Entrada | Tensor float32 (B, 3, 224, 224), normalizado con media (0.485, 0.456, 0.406) y desviación (0.229, 0.224, 0.225) |
| Salidas | $p$ (B, 6), $\hat\sigma$ (B,), $\pi$ (B,), $p_V$ (B, 6), $p_C$ (B, 6) |
| Formatos | ONNX (opset 17) y TorchScript; ida y vuelta < $10^{-5}$ |
| Licencia | MIT (código); pesos derivados de modelos de torchvision preentrenados en ImageNet |

## Uso previsto
Investigación y docencia sobre clasificación de residuos y robustez al ruido. **No** está validado para separar residuos en planta sin una nueva evaluación en ese dominio.

## Datos de entrenamiento
TrashNet (Yang y Thung, 2016), partición v2 agrupada por casi duplicados: 1800 imágenes de entrenamiento y 360 de validación. Ver `DATA_STATEMENT.md`.

## Desempeño (prueba, 361 imágenes; media sobre 20 semillas, IC 95 %)
| Métrica | NAGFuse | Mejor rival |
|---|---|---|
| F1 macro limpio | 0.936 [0.931, 0.942] | RWCNet 0.937 (equivalente, TOST $p < 0.001$) |
| F1 macro, ruido gaussiano $\sigma = 0.05$ | 0.904 | ViT-B16 0.892 (n.s.) |
| F1 macro, ruido $\sigma = 0.10$ | 0.859 | ViT-B16 0.840 |
| ECE | 0.030 | RWCNet 0.030 |
| F1 macro RealWaste / Garbage v2 / WaRP-C (sin reentrenar) | 0.46 / 0.52 / 0.36 | ViT-B16 0.52 / 0.59 / 0.44 |

## Limitaciones y riesgos
- **Fuera de dominio.** Fuera de fotos de un objeto sobre fondo claro, el F1 cae a 0.36–0.52, y es peor que el de ViT-B16 solo.
- **El estimador asume fondos lisos.** Con fondos texturizados $\hat\sigma$ puede confundir textura con ruido.
- **Ruido evaluado.** Solo ruido gaussiano aditivo sintético.
- **Clase trash.** Es la peor cubierta: su cobertura conforme al 90 % es de 0.76.
- **Costo.** Es costoso: dos redes, unas 18 GMAC por imagen.

## Consideraciones éticas
Errores de clasificación en reciclaje pueden contaminar lotes. Se recomienda usar el conjunto conforme y los avisos de incertidumbre del dashboard, y revisar manualmente los casos inciertos.
