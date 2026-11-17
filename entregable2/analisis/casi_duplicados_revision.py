import pandas as pd, numpy as np
from comun import OUT, MODELS, SEEDS, aligned, fast_f1_macro
# Revisión visual de los pares (mismo objeto: si / no / incierto), registrada tras inspeccionar las imágenes
rev = [("paper/paper148.jpg","si"),("paper/paper549.jpg","si"),("metal/metal122.jpg","si"),("paper/paper53.jpg","si"),
       ("paper/paper284.jpg","si"),("paper/paper467.jpg","si"),("cardboard/cardboard27.jpg","no"),("plastic/plastic403.jpg","si"),
       ("metal/metal361.jpg","si"),("plastic/plastic43.jpg","si"),("metal/metal31.jpg","si"),("plastic/plastic26.jpg","si"),
       ("cardboard/cardboard393.jpg","no"),("glass/glass419.jpg","si"),("glass/glass297.jpg","si"),("paper/paper208.jpg","si"),
       ("glass/glass437.jpg","si"),("plastic/plastic323.jpg","si"),("plastic/plastic329.jpg","si"),
       # muestra aleatoria (semilla 1) de 18 pares con similitud entre 0.80 y 0.88
       ("glass/glass391.jpg","si"),("glass/glass95.jpg","si"),("cardboard/cardboard318.jpg","no"),("glass/glass491.jpg","no"),
       ("cardboard/cardboard133.jpg","incierto"),("glass/glass131.jpg","si"),("glass/glass289.jpg","no"),("glass/glass178.jpg","si"),
       ("glass/glass206.jpg","si"),("metal/metal341.jpg","incierto"),("cardboard/cardboard160.jpg","no"),("paper/paper485.jpg","incierto"),
       ("metal/metal140.jpg","incierto"),("cardboard/cardboard57.jpg","no"),("metal/metal241.jpg","incierto"),("paper/paper154.jpg","incierto"),
       ("plastic/plastic37.jpg","si"),("glass/glass487.jpg","incierto")]
emb = pd.read_csv(OUT / "casi_duplicados_embeddings.csv")
r = emb.merge(pd.DataFrame(rev, columns=["prueba", "mismo_objeto"]), on="prueba", how="inner")
r["franja"] = np.where(r.similitud >= 0.88, ">= 0.88 (todos revisados)", "0.80-0.88 (muestra de 18)")
r.to_csv(OUT / "casi_duplicados_revision.csv", index=False)
print(r.groupby(["franja", "mismo_objeto"]).size())
# Sensibilidad: F1 sin las imágenes de prueba por encima de cada umbral de similitud
rows = []
for nombre, excl in [("todas (361)", set()), ("sin sim >= 0.88", set(emb[emb.similitud >= 0.88].prueba)),
                     ("sin sim >= 0.80", set(emb[emb.similitud >= 0.80].prueba))]:
    for m in MODELS:
        f = []
        for s in SEEDS:
            d = aligned(m, s); k = ~np.isin(d["relpath"], list(excl))
            f.append(fast_f1_macro(d["y_true"][k], d["y_pred"][k]))
        rows.append({"conjunto": nombre, "n_imagenes": 361 - len(excl), "model": m, "f1_media": np.mean(f), "f1_de": np.std(f, ddof=1)})
sen = pd.DataFrame(rows)
sen["rango"] = sen.groupby("conjunto").f1_media.rank(ascending=False)
sen.to_csv(OUT / "casi_duplicados_sensibilidad.csv", index=False)
print(sen.pivot(index="model", columns="conjunto", values="f1_media").round(3).sort_values("todas (361)", ascending=False))
