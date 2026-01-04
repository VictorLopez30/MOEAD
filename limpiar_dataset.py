import pandas as pd

INPUT = "dataset.csv"
OUTPUT = "discretized_dataset.csv"

DROP_COLS = ["r1","r2","r3","r4","r5","r6","r7","r8","r9","r10","r11","r12","r13","r14","r15","r16","r17","r18","r19","r20","r21","r22","r23","r24","r25","r26","r27"]

QUINTILE_COLS = ["freq ceros","freq unos","freq dos","width","actividad","entropia","sensitividad","time entropy"]


def _to_quintiles(series: pd.Series) -> pd.Series:
    bins = pd.qcut(series, q=5, labels=False, duplicates="drop")
    return bins.add(1)


def discretize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in QUINTILE_COLS:
        if col in df.columns:
            df[col] = _to_quintiles(df[col])

    if "pasos" in df.columns:
        df["pasos"] = df["pasos"].apply(lambda x: 1 if x == -1 else 2)

    if "simetria" in df.columns:
        df["simetria"] = df["simetria"].apply(lambda x: 1 if bool(x) else 2)

    if "transitoriedad" in df.columns:
        df["transitoriedad"] = df["transitoriedad"].apply(lambda x: 2 if x == 1001 else 1)

    return df


def main() -> None:
    df = pd.read_csv(INPUT)
    df.drop(columns=DROP_COLS, inplace=True, errors="ignore")
    df = discretize(df)
    df.to_csv(OUTPUT, index=False, encoding="utf-8")

    print(f"Archivo guardado: {OUTPUT}")
    print(f"Filas: {len(df)} | Columnas: {list(df.columns)}")


if __name__ == "__main__":
    main()
