# 27. Protocolo y validación estadística

## 27.1 Preregistro

| Documento | Contenido | Fecha y tag |
|---|---|---|
| `estandar.md` | Estándar de profundidad, rivales, conjuntos adicionales y **hallazgo alternativo** si NAGFuse no supera a toda la familia | 2026-11-21, `e3-protocolo-v1` |
| `protocolo.md` | Hipótesis, predicciones, familias de contrastes, pruebas, semillas, potencia y regla de corte | 2026-11-21, `e3-protocolo-v1` |
| `design_log.md` | Bitácora de desarrollo ($N_d = 7$) y enmiendas | v1, v2 (22-nov) y v3 (22-nov) |

El tag `e3-protocolo-v1` se creó antes de la primera corrida definitiva. Hubo dos enmiendas, ambas declaradas en `design_log.md` antes de abrir cualquier métrica de prueba:

- **v2: corrección técnica.** Logits no finitos en la variante sin contracción; solo afectó a la ablación.
- **v3: adelanto de la hora de corte.** Se pasó del lunes a las 06:00 a la noche del domingo, porque las 144 tareas ya habían terminado. El conjunto de corridas al corte es el mismo a cualquier hora.

**Exposición declarada.** Los registros de entrenamiento del código congelado del E2 imprimen el F1 de prueba de cada corrida. No se usaron para ninguna decisión: todas las decisiones de diseño son anteriores a v1, y después de v1 no se cambió ningún hiperparámetro, rival, conjunto ni métrica.

## 27.2 Pruebas

- **Familia principal (m = 12).** NAGFuse frente a cada rival, en F1 macro de prueba.
- **Tres pruebas por contraste:**
  1. t pareada sobre semillas;
  2. Wilcoxon de rangos con signo exacta;
  3. bootstrap pareado jerárquico (10 000 réplicas): se remuestrean semillas y, dentro de cada semilla, los grupos de casi duplicados del conjunto de prueba.
- **Holm** por prueba. **Decide el máximo de los tres p ajustados.**
- **Criterio para "supera":** p ajustado < 0.05, diferencia media > $\delta = 0.02$ y límite inferior del IC 95 % > 0.
- **Baselines deterministas.** HistGradientBoosting y el trivial se tratan como constantes.

El p del bootstrap tiene una resolución de $10^{-4}$, de modo que su valor ajustado mínimo es 0.0012.

## 27.3 H1: F1 macro limpio

| Rival | R | Diferencia | IC 95 % | p t (Holm) | p Wilcoxon (Holm) | p bootstrap (Holm) | δ de Cliff | $d_z$ | Decisión |
|---|---|---|---|---|---|---|---|---|---|
| Celik-HC | 20 | +0.0272 | [0.0168, 0.0392] | <0.001 | <0.001 | 0.001 | 0.90 | 1.49 | Supera (> δ) |
| MbXception | 12 | +0.0692 | [0.0566, 0.0832] | <0.001 | 0.004 | 0.001 | 1.00 | 6.07 | Supera (> δ) |
| ViT-B16 | 20 | +0.0002 | [-0.0110, 0.0118] | 1.000 | 1.000 | 1.000 | 0.10 | 0.01 | No significativa |
| GFN | 20 | +0.0125 | [0.0030, 0.0224] | 0.006 | 0.005 | 0.043 | 0.50 | 0.81 | Mejor, < δ |
| FocusRCNet | 12 | +0.1260 | [0.1088, 0.1462] | <0.001 | 0.004 | 0.001 | 1.00 | 5.02 | Supera (> δ) |
| ResMsCapsule | 20 | +0.0308 | [0.0197, 0.0427] | <0.001 | <0.001 | 0.001 | 0.80 | 1.57 | Supera (> δ) |
| RWCNet | 20 | -0.0004 | [-0.0096, 0.0088] | 1.000 | 1.000 | 1.000 | -0.10 | -0.03 | No significativa |
| CNN | 12 | +0.1110 | [0.0956, 0.1283] | <0.001 | 0.004 | 0.001 | 1.00 | 5.74 | Supera (> δ) |
| MobileNetV2 | 12 | +0.1291 | [0.1123, 0.1477] | <0.001 | 0.004 | 0.001 | 1.00 | 5.73 | Supera (> δ) |
| ResNet-18 | 12 | +0.0713 | [0.0572, 0.0870] | <0.001 | 0.004 | 0.001 | 1.00 | 4.23 | Supera (> δ) |
| HistGradientBoosting | 20 | +0.1763 | [0.1673, 0.1836] | <0.001 | <0.001 | 0.001 | 1.00 | 15.85 | Supera (> δ) |
| Trivial | 20 | +0.8729 | [0.8639, 0.8802] | <0.001 | <0.001 | 0.001 | 1.00 | 78.49 | Supera (> δ) |

```{figure} images/e3/forest_H1_H2.png
:width: 100%
Diferencias pareadas NAGFuse − rival con IC 95 % del bootstrap jerárquico (eje simétrico-logarítmico). La línea discontinua es $\delta = 0.02$.
```

**Resultado.** NAGFuse supera a 10 de los 12 rivales con diferencia mayor que $\delta$. Frente a GFN la ventaja es significativa pero menor que $\delta$ (+0.0125). Frente a RWCNet y ViT-B16 no hay diferencia. **La condición de superioridad sobre toda la familia no se cumple.**

**Potencia observada.** Con las desviaciones pareadas observadas y 20 semillas, la potencia para detectar $\delta$ al nivel 0.05/12 es de 0.99 frente a RWCNet ($\sigma_d = 0.015$) y de 0.83 frente a ViT-B16 ($\sigma_d = 0.021$). Estas comparaciones podían detectar $\delta$; el empate es informativo y no se debe a falta de semillas. Las diferencias mínimas detectables son 0.016 y 0.021. Frente a FocusRCNet, MobileNetV2 y CNN (12 semillas, $\sigma_d$ entre 0.019 y 0.025) la potencia para $\delta$ es baja, pero las diferencias observadas superan 0.1.

## 27.4 Sensibilidad de la conclusión

| Rival | F1, tres pruebas | t de Nadeau–Bengio | MCC | Semillas 0–9 |
|---|---|---|---|---|
| Celik-HC | Supera | Supera | Supera | < δ |
| MbXception | Supera | Supera | Supera | Supera |
| ViT-B16 | n.s. | n.s. | n.s. | n.s. |
| GFN | < δ | n.s. | n.s. | n.s. |
| FocusRCNet | Supera | Supera | Supera | Supera |
| ResMsCapsule | Supera | Supera | Supera | Supera |
| RWCNet | n.s. | n.s. | n.s. | n.s. |
| CNN | Supera | Supera | Supera | Supera |
| MobileNetV2 | Supera | Supera | Supera | Supera |
| ResNet-18 | Supera | Supera | Supera | Supera |
| HistGradientBoosting | Supera | Supera | Supera | Supera |
| Trivial | Supera | Supera | Supera | Supera |

La conclusión sobre RWCNet y ViT-B16 (sin diferencia) no cambia con ninguna variante: la prueba, la métrica o el subconjunto de semillas. Frente a GFN, la ventaja deja de ser significativa con la t corregida, con MCC y con 10 semillas, así que es frágil. Frente a Celik-HC, con solo 10 semillas la diferencia (0.0199) queda justo por debajo de $\delta$.

## 27.5 H3: no inferioridad frente al mejor rival

El mejor rival en F1 limpio es RWCNet (0.9369, 20 semillas). Diferencia pareada: $-0.0004$, IC 90 % $[-0.0063, 0.0054]$. Está dentro de $\pm\delta$, con TOST $p = 7\times10^{-6}$. **NAGFuse es equivalente al mejor modelo del benchmark.**

## 27.6 H2: ruido gaussiano $\sigma = 0.05$

| Rival | R | Diferencia | IC 95 % | p t (Holm) | p Wilcoxon (Holm) | p bootstrap (Holm) | δ de Cliff | $d_z$ | Decisión |
|---|---|---|---|---|---|---|---|---|---|
| Celik-HC | 20 | +0.6602 | [0.6165, 0.7011] | <0.001 | <0.001 | 0.001 | 1.00 | 6.95 | Supera (> δ) |
| MbXception | 12 | +0.6797 | [0.6451, 0.7139] | <0.001 | 0.003 | 0.001 | 1.00 | 11.85 | Supera (> δ) |
| ViT-B16 | 20 | +0.0121 | [-0.0009, 0.0258] | 0.034 | 0.021 | 0.066 | 0.50 | 0.51 | No significativa |
| GFN | 20 | +0.1203 | [0.0921, 0.1517] | <0.001 | <0.001 | 0.001 | 1.00 | 1.85 | Supera (> δ) |
| FocusRCNet | 12 | +0.7611 | [0.7274, 0.7937] | <0.001 | 0.003 | 0.001 | 1.00 | 13.47 | Supera (> δ) |
| ResMsCapsule | 20 | +0.4767 | [0.4403, 0.5154] | <0.001 | <0.001 | 0.001 | 1.00 | 5.78 | Supera (> δ) |
| RWCNet | 20 | +0.3077 | [0.2712, 0.3430] | <0.001 | <0.001 | 0.001 | 1.00 | 3.88 | Supera (> δ) |
| CNN | 12 | +0.7216 | [0.7024, 0.7440] | <0.001 | 0.003 | 0.001 | 1.00 | 18.63 | Supera (> δ) |
| MobileNetV2 | 12 | +0.6184 | [0.5922, 0.6494] | <0.001 | 0.003 | 0.001 | 1.00 | 11.71 | Supera (> δ) |
| ResNet-18 | 12 | +0.5073 | [0.4884, 0.5260] | <0.001 | 0.003 | 0.001 | 1.00 | 14.74 | Supera (> δ) |

Los dos baselines deterministas no tienen evaluación con ruido en el E2, así que la familia de H2 queda en m = 10. Para los modelos del E1 no se guardaron predicciones por imagen con ruido, y en esos casos el bootstrap remuestrea solo semillas.

**Resultado.** NAGFuse supera a todas las CNN de la familia con márgenes de 0.12 (GFN) a 0.76 (FocusRCNet). Frente a ViT-B16 la ventaja es de +0.012, no significativa y menor que $\delta$. **La hipótesis del mecanismo, tal como se preregistró (mejorar frente a cada CNN en al menos $\delta$ sin perder más de $\delta$ en limpio), se confirma.**

## 27.7 Comparación global

Friedman con la corrección de Iman–Davenport sobre los 11 modelos profundos (semillas 0–11): $\chi^2 = 106.9$, $F_{ID} = 89.8$, $p < 10^{-40}$. Los rangos medios son RWCNet 2.00, ViT-B16 2.25, **NAGFuse 2.83**, GFN 3.75 y Celik-HC 5.00. La diferencia crítica de Nemenyi es 4.36, de modo que los cinco primeros no se distinguen entre sí con este procedimiento, que es más conservador que los contrastes pareados.

```{figure} images/e3/diferencia_critica.png
:width: 100%
Diagrama de diferencia crítica. Las barras unen modelos cuya diferencia de rango medio no supera la diferencia crítica.
```
