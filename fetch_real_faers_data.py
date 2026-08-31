import json
import urllib.error
import urllib.request
import urllib.parse
import pandas as pd
from datetime import datetime

HTTP_TIMEOUT = 30  # seconds


def fetch_openfda_reports(limit: int = 500, query_drug: str = "atorvastatin") -> pd.DataFrame:
    """
    Fetches real spontaneous ADR reports from the openFDA Drug Event API endpoint.
    API documentation: https://open.fda.gov/apis/drug/event/
    """
    base_url = "https://api.fda.gov/drug/event.json"
    search_query = f'patient.drug.medicinalproduct:"{query_drug}"'
    params = {
        "search": search_query,
        "limit": min(limit, 1000)
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    print(f"Fetching {limit} real reports from openFDA API for query: '{query_drug}'...")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (ADR-Duplicate-Resolution-Research)"}
    )

    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # openFDA returns 404 for "no results" and 429 when rate-limited.
        detail = ""
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("error", {}).get("message", "")
        except Exception:
            pass
        raise SystemExit(f"openFDA request failed ({exc.code} {exc.reason}). {detail}".strip())
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not reach openFDA: {exc.reason}")

    results = payload.get("results", [])
    print(f"Successfully retrieved {len(results)} raw records.")

    records = []
    base_epoch = datetime(2015, 1, 1)

    for item in results:
        report_id = item.get("safetyreportid", "")
        if not report_id:
            continue

        patient = item.get("patient", {})

        # 1. Parse Patient Age (convert units to years if available)
        age = None
        raw_age = patient.get("patientonsetage")
        raw_unit = str(patient.get("patientonsetageunit", "801"))  # 801 = Years, 802 = Months, 804 = Days
        if raw_age:
            try:
                age_val = float(raw_age)
                if raw_unit == "801":
                    age = age_val
                elif raw_unit == "802":
                    age = round(age_val / 12.0, 1)
                elif raw_unit == "804":
                    age = round(age_val / 365.25, 1)
                else:
                    age = age_val
                if not (0 <= age <= 120):
                    age = None
            except ValueError:
                age = None

        # 2. Parse Patient Sex (1 = Male, 2 = Female, 0/other = Unknown)
        raw_sex = str(patient.get("patientsex", ""))
        sex_map = {"1": "Male", "2": "Female"}
        sex = sex_map.get(raw_sex, "")

        # 3. Parse Seriousness (1 = Serious, 2 = Non-serious)
        raw_ser = str(item.get("serious", ""))
        seriousness = "Serious" if raw_ser == "1" else "Non-serious"

        # 4. Parse Report Date into relative Day Count
        receipt_str = item.get("receiptdate") or item.get("receivedate")
        if receipt_str:
            try:
                dt = datetime.strptime(receipt_str, "%Y%m%d")
                report_day = (dt - base_epoch).days
            except ValueError:
                report_day = 0
        else:
            report_day = 0

        # 5. Extract Medicinal Products / Active Substances
        drugs = set()
        for d in patient.get("drug", []):
            med_name = d.get("medicinalproduct", "")
            if med_name:
                drugs.add(med_name.strip())
            # Also capture generic name if present
            openfda = d.get("openfda", {})
            for gen in openfda.get("generic_name", []):
                if gen:
                    drugs.add(gen.strip())

        # 6. Extract MedDRA Reaction Preferred Terms (PT)
        events = set()
        for r in patient.get("reaction", []):
            pt = r.get("reactionmeddrapt", "")
            if pt:
                events.add(pt.strip())

        if drugs and events:
            records.append({
                "ReportID": str(report_id),
                "Age": age if age is not None else "",
                "Sex": sex,
                "Seriousness": seriousness,
                "ReportDay": report_day,
                "Drugs": ";".join(sorted(drugs)),
                "Events": ";".join(sorted(events))
            })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df_faers = fetch_openfda_reports(limit=500, query_drug="atorvastatin")
    output_filename = "real_faers_sample.csv"
    df_faers.to_csv(output_filename, index=False)
    print(f"\nSaved {len(df_faers)} cleaned FAERS records to '{output_filename}'.")
    print(df_faers.head(3))