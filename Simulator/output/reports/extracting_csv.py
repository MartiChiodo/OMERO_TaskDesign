import os
import re
from glob import glob

import pandas as pd


# =============================================================================
# Cartelle
# =============================================================================

REPORT_FOLDER = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Pattern nomi file
# =============================================================================

filename_pattern = re.compile(
    r"report_(.+?)_Opt.+?_Seed(\d+)\.txt"
)


# =============================================================================
# Pattern tabella ordini
# =============================================================================

order_pattern = re.compile(
    r"^\s*(\d+)\s+\d+\s+([\d\.]+)\s*$"
)

total_pattern = re.compile(
    r"^\s*Total\s+\d+\s+([\d\.]+)\s*$"
)


# =============================================================================
# Funzione estrazione dati
# =============================================================================

def extract_reports(mode_folder):

    avg_pods = {}
    comp_time = {}
    throughput = {}
    flow_data = {}


    files = glob(
        os.path.join(
            mode_folder,
            "report_*_Opt*_Seed*.txt"
        )
    )

    print(f"\nAnalizzo {mode_folder}")
    print(f"Report trovati: {len(files)}")


    for filepath in files:

        filename = os.path.basename(filepath)

        m = filename_pattern.match(filename)

        if not m:
            print("Nome non riconosciuto:", filename)
            continue


        scenario = m.group(1)
        seed = int(m.group(2))


        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()


        # -------------------------------------------------------------
        # Metriche principali
        # -------------------------------------------------------------

        for line in lines:


            # Throughput
            if "Total number of items picked" in line:

                value = float(
                    line.split("=")[1]
                    .strip()
                    .rstrip(".")
                )

                throughput.setdefault(
                    scenario, {}
                )[seed] = value



            # Average pods moving

            elif "Average number of pod moving simultaneously" in line:

                value = float(
                    line.split("=")[1]
                    .strip()
                    .rstrip(".")
                )

                avg_pods.setdefault(
                    scenario, {}
                )[seed] = value



            # Computational time

            elif "Computational time spent for making decisions" in line:

                value = float(
                    line.split("=")[1]
                    .replace("sec.", "")
                    .strip()
                )

                comp_time.setdefault(
                    scenario, {}
                )[seed] = value



        # -------------------------------------------------------------
        # Mean flow time
        # -------------------------------------------------------------

        inside_table = False


        for line in lines:


            if "ORDERS BY SIZE" in line:
                inside_table = True
                continue


            if not inside_table:
                continue


            # Riga Total

            m_total = total_pattern.match(line)

            if m_total:

                avg_flow = float(m_total.group(1))

                flow_data.setdefault(
                    (scenario, "Total"), {}
                )[seed] = avg_flow

                break



            # Riga ordine

            m_order = order_pattern.match(line)

            if m_order:

                order_size = int(m_order.group(1))
                avg_flow = float(m_order.group(2))


                flow_data.setdefault(
                    (scenario, order_size), {}
                )[seed] = avg_flow


    return (
        avg_pods,
        comp_time,
        throughput,
        flow_data
    )



# =============================================================================
# Salvataggio CSV
# =============================================================================

def save_matrix_csv(data, filename):

    df = pd.DataFrame.from_dict(
        data,
        orient="index"
    )

    df.index.name = "Scenario"

    df = df.sort_index()

    df = df.reindex(
        sorted(df.columns),
        axis=1
    )

    df.to_csv(
        os.path.join(
            REPORT_FOLDER,
            filename
        )
    )



def save_flow_csv(flow_data, filename):

    rows = []


    for (scenario, order_size), values in flow_data.items():

        row = {
            "Scenario": scenario,
            "OrderSize": order_size
        }

        row.update(values)

        rows.append(row)


    df = pd.DataFrame(rows)


    if not df.empty:

        seed_cols = sorted(
            [
                c for c in df.columns
                if isinstance(c, int)
            ]
        )


        df = df[
            ["Scenario", "OrderSize"] + seed_cols
        ]


        def sort_key(x):

            if x == "Total":
                return 9999

            return int(x)


        df = df.sort_values(
            by=["Scenario", "OrderSize"],
            key=lambda col:
                col.map(sort_key)
                if col.name == "OrderSize"
                else col
        )


        df.to_csv(
            os.path.join(
                REPORT_FOLDER,
                filename
            ),
            index=False
        )



# =============================================================================
# MAIN
# =============================================================================

for mode in ["Opt_False", "Opt_True"]:

    mode_folder = os.path.join(
        REPORT_FOLDER,
        mode
    )


    if not os.path.exists(mode_folder):

        print(
            f"Cartella {mode_folder} non trovata"
        )

        continue



    (
        avg_pods,
        comp_time,
        throughput,
        flow_data

    ) = extract_reports(mode_folder)



    save_matrix_csv(
        avg_pods,
        f"{mode}_average_pods.csv"
    )


    save_matrix_csv(
        comp_time,
        f"{mode}_computational_time.csv"
    )


    save_matrix_csv(
        throughput,
        f"{mode}_throughput.csv"
    )


    save_flow_csv(
        flow_data,
        f"{mode}_mean_flow_time.csv"
    )


print("\nCSV creati correttamente.")