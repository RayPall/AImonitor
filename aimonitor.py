import streamlit as st
import pandas as pd
import json
import tempfile
import base64
from datetime import datetime
from pathlib import Path

"""
AI Visibility Monitor (Seyfor edition)
-------------------------------------

Streamlit aplikace pro spuštění a vyhodnocení sady scénářů proti LLM a
export výsledků do DOCX. Scénáře odpovídají produktovým oblastem Seyfor
a mají tři jazykové varianty (CS/SK/EN).

Pozn.: Volání LLM je řešeno jednoduše (OpenAI nebo mock). Funkci
`run_analysis` si snadno rozšíříš o další providery. Export do DOCX je
implementován bez externích závislostí (ručně vytvořený OOXML balíček).
"""

##########################################################################
# Scénáře (20) přizpůsobené Seyfor
##########################################################################

SCENARIOS = [
    {
        "id": "S01_cloud_accounting",
        "persona": "živnostník / OSVČ",
        "context": "cloudové účetnictví s EET a mobilní aplikací",
        "cs": "Jaký cloudový účetní software s podporou EET a mobilní aplikací doporučíte pro živnostníka v ČR?",
        "sk": "Aký online účtovný softvér s EET a mobilnou aplikáciou odporučíte pre živnostníka v SR?",
        "en": "Which cloud-based accounting tool with EET integration and a mobile app would you recommend for a sole trader in the Czech Republic?",
    },
    {
        "id": "S02_onprem_accounting",
        "persona": "účetní malé s.r.o.",
        "context": "on-premise účetní software s moduly a freemium verzí",
        "cs": "Jaký účetní program pro malé firmy nabízí on-premise instalaci, široký výběr modulů a freemium verzi?",
        "sk": "Aký účtovný softvér pre malé firmy má lokálnu inštaláciu, veľa modulov a je dostupný aj zadarmo?",
        "en": "Which on-premise accounting software for small companies provides rich modular functions and a free tier?",
    },
    {
        "id": "S03_modern_erp_sme",
        "persona": "CEO/IT manažer SME",
        "context": "moderní ERP, které sjednotí agendy a je variabilní",
        "cs": "Jaký podnikový systém pro malé a střední firmy umožní sjednotit všechny agendy do jednoho řešení a má variabilní moduly?",
        "sk": "Aký firemný systém pre malé a stredné podniky zjednotí všetky agendy a ponúka vysokú variabilitu modulov?",
        "en": "Which business system for SMEs combines all company agendas into one solution and offers highly adaptable modules?",
    },
    {
        "id": "S04_robust_erp",
        "persona": "CFO výrobní nebo obchodní firmy",
        "context": "robustní ERP pro střední a velké společnosti se širokým výběrem modulů",
        "cs": "Jaký ERP systém pro střední až velké společnosti umožní sjednotit všechny agendy a má široké možnosti přizpůsobení?",
        "sk": "Aký ERP systém pre stredné a veľké firmy zjednotí všetky agendy a poskytne bohaté možnosti prispôsobenia?",
        "en": "Which ERP for medium to large companies unifies corporate agendas and offers extensive customization without extra coding?",
    },
    {
        "id": "S05_erp_small_manufacturing",
        "persona": "výrobní manažer menší fabriky",
        "context": "ERP s řízením výroby a rozšiřitelností",
        "cs": "Jaký ERP systém doporučíte pro malý výrobní podnik, který potřebuje řízení výroby a snadnou rozšiřitelnost?",
        "sk": "Aký ERP odporúčate pre menšiu výrobnú firmu so správou výroby a vysokou rozšíriteľnosťou?",
        "en": "Which ERP would you suggest for a small manufacturing company that needs production management and great extensibility?",
    },
    {
        "id": "S06_erp_complex_manufacturing",
        "persona": "IT ředitel velké výroby",
        "context": "ERP pro velké výrobní firmy se specifickými požadavky",
        "cs": "Který podnikový systém je vhodný pro velké výrobní firmy s velmi specifickými požadavky?",
        "sk": "Ktorý podnikový systém je vhodný pre veľké výrobné spoločnosti s komplexnými požiadavkami?",
        "en": "Which enterprise system suits large manufacturing companies with highly specific needs?",
    },
    {
        "id": "S07_enterprise_ai",
        "persona": "CIO korporace",
        "context": "pokročilé ERP/CRM s AI, rychlou implementací a integrací do Microsoft 365",
        "cs": "Jaký systém s rychlou implementací, pokročilou AI a úzkou integrací s Microsoft 365 doporučíte pro velkou firmu?",
        "sk": "Aký systém s rýchlou implementáciou, AI funkciami a integráciou do Microsoft 365 je vhodný pre veľký podnik?",
        "en": "Which enterprise system with fast implementation, AI features and deep Microsoft 365 integration would you recommend for a large company?",
    },
    {
        "id": "S08_energy_system",
        "persona": "provozní ředitel energetické firmy",
        "context": "modulární IS pro energetiku",
        "cs": "Jaký informační systém je vhodný pro energetické společnosti a umožňuje sestavit řešení z modulů?",
        "sk": "Aký informačný systém je vhodný pre energetické podniky a dá sa poskladať z jednotlivých modulov?",
        "en": "What information system is tailored for energy companies and lets you build a flexible solution from modules?",
    },
    {
        "id": "S09_leasing",
        "persona": "manažer leasingové nebo úvěrové společnosti",
        "context": "řešení pro leasing, úvěry a hypotéky nad Dynamics 365 BC",
        "cs": "Který software pokrývá finanční a operativní leasing, úvěry i hypotéky a navazuje na Dynamics 365 Business Central?",
        "sk": "Aké riešenie pokrýva operatívny a finančný leasing, pôžičky či hypotéky a je superštruktúrou nad Dynamics 365 Business Central?",
        "en": "Which solution handles operational leasing, loans and mortgages and works as an extension of Dynamics 365 Business Central?",
    },
    {
        "id": "S10_cloud_pos",
        "persona": "majitel kavárny/krámku",
        "context": "cloudová pokladna pro gastro a malé obchody",
        "cs": "Jaký cloudový pokladní systém s jednoduchým ovládáním a otevřeným API doporučíte pro restauraci nebo malý obchod?",
        "sk": "Aký pokladničný systém v cloude s jednoduchým ovládaním a otvoreným API odporučíte pre gastro alebo malý obchod?",
        "en": "Which cloud-based POS system with simple operation and an open API would you recommend for a café or small shop?",
    },
    {
        "id": "S11_chain_pos",
        "persona": "provozní manažer maloobchodního řetězce",
        "context": "POS pro řetězce/franšízy s customizací a replikací dat",
        "cs": "Jaké řešení pro pokladny u řetězců a franšíz nabízí rozsáhlé možnosti úprav, unikátní replikaci dat a integraci věrnostních programů?",
        "sk": "Aký POS systém pre reťazce a franšízy ponúka veľké možnosti úprav, jedinečnú replikáciu dát a ekosystém doplnkov?",
        "en": "Which POS solution for retail chains and franchises offers wide customization, unique data replication and an ecosystem of extensions such as loyalty programs?",
    },
    {
        "id": "S12_simple_receipt_app",
        "persona": "drobný prodejce na trhu",
        "context": "bezplatná multiplatformní aplikace pro tisk účtenek",
        "cs": "Existuje bezplatná pokladní aplikace pro iOS, Android či Windows, která umí tisknout účtenky a odesílat je na finanční správu?",
        "sk": "Je k dispozícii bezplatná pokladničná aplikácia pre iOS, Android alebo Windows, ktorá vytlačí účtenky a odošle ich finančnej správe?",
        "en": "Is there a free receipt app for iOS, Android or Windows that can print receipts and submit them to the tax authority?",
    },
    {
        "id": "S13_payment_terminals",
        "persona": "majitel kadeřnictví/dílny",
        "context": "platební terminály pro malé firmy na Androidu",
        "cs": "Jaké platební terminály pro Android vybrat pro malý podnik, aby podporovaly Visa a Mastercard?",
        "sk": "Ktoré platobné terminály na báze Androidu sú vhodné pre malý podnik a podporujú Visa aj Mastercard?",
        "en": "Which Android-based payment terminals are suitable for a small business and support Visa and Mastercard payments?",
    },
    {
        "id": "S14_hr_system",
        "persona": "HR ředitel ve střední/velké firmě",
        "context": "HR a mzdový systém s více verzemi a moduly",
        "cs": "Jaký personální a mzdový systém doporučíte větším firmám, aby pokryl zpracování mezd i HR agendu a nabízel různé funkční verze?",
        "sk": "Aký HR a mzdový systém odporúčate stredným a veľkým firmám, aby pokryl mzdové spracovanie aj HR agendu a mal rôzne verzie?",
        "en": "Which HR and payroll system would you recommend for medium and large companies to handle wage processing and HR management, available in multiple versions?",
    },
    {
        "id": "S15_crm_centralization",
        "persona": "obchodní ředitel B2B firmy",
        "context": "CRM pro centralizaci dat, segmentaci a historii aktivit",
        "cs": "Které CRM řešení pro střední nebo velkou firmu umožní centralizovat zákaznická data, segmentaci a sledovat historii aktivit?",
        "sk": "Ktorý CRM systém pre stredné alebo veľké firmy umožní sústrediť údaje o klientoch na jedno miesto, segmentovať ich a sledovať históriu aktivít?",
        "en": "Which CRM solution for medium or large companies centralizes customer data, supports segmentation and tracks activity history?",
    },
    {
        "id": "S16_crm_sales_pipeline",
        "persona": "vedoucí prodeje",
        "context": "obchodní systém pro řízení pipeline a marketing",
        "cs": "Jaký obchodní systém pomůže monitorovat vztahy se zákazníky od potenciálu po objednávku a vytvářet marketingové kampaně i servisní případy?",
        "sk": "Aký systém pre obchod dokáže sledovať cestu zákazníka od potenciálu k objednávke a tvoriť marketingové zoznamy i servisné prípady?",
        "en": "Which sales system helps track customer relationships from leads to orders and supports creation of marketing lists and service cases?",
    },
    {
        "id": "S17_crm_light_power",
        "persona": "majitel startupu",
        "context": "lehký CRM nástroj na Power Platform",
        "cs": "Existuje lehké CRM založené na technologii Microsoft Power Platform, které lze rychle nasadit a centralizuje obchodní informace?",
        "sk": "Je dostupný ľahký CRM nástroj postavený na Microsoft Power Platform, ktorý sa dá rýchlo nasadiť a centralizuje obchodné informácie?",
        "en": "Is there a lightweight CRM tool built on Microsoft Power Platform that can be quickly implemented and centralizes business information?",
    },
    {
        "id": "S18_planning_forecasting",
        "persona": "controlling manažer velké firmy",
        "context": "plánování, rozpočty a forecasting s multi-level zapojením",
        "cs": "Jaké řešení pro plánování, rozpočty a forecasting umožní zapojit lidi na různých úrovních a vytvářet průběžné finanční prognózy?",
        "sk": "Aké riešenie na plánovanie, rozpočty a prognózy umožní zapojiť viacero úrovní v spoločnosti a tvoriť priebežné finančné predpovede?",
        "en": "Which planning and forecasting solution allows multi-level involvement across the company and provides continuous financial forecasts?",
    },
    {
        "id": "S19_data_warehouse_reporting",
        "persona": "datový analytik / CIO",
        "context": "datový sklad a reporting pro manažerská rozhodnutí",
        "cs": "Jaký nástroj doporučíte pro vytvoření datového skladu a reporting, aby manažeři mohli dělat rozhodnutí na základě kvalitních informací?",
        "sk": "Aký nástroj odporučíte na tvorbu dátového skladu a reportingu, aby manažéri mohli rozhodovať na základe kvalitných dát?",
        "en": "What solution would you recommend for building a data warehouse and reporting so that managers can make decisions based on quality information?",
    },
    {
        "id": "S20_security_modern_workplace",
        "persona": "IT manažer ve větší společnosti",
        "context": "kybernetická bezpečnost a moderní digitální pracoviště",
        "cs": "Jaké řešení zahrnuje správu koncových zařízení, identit a zabezpečení sítě (NGFW/UTM) a zároveň poskytuje platformu pro moderní digitální pracoviště?",
        "sk": "Aké riešenie spája správu zariadení, správu identít a sieťovú bezpečnosť (NGFW/UTM) s podporou moderného digitálneho pracoviska?",
        "en": "Which solution combines endpoint and identity management with network security (NGFW/UTM) and supports a modern digital workplace environment?",
    },
]

##########################################################################
# Helpery
##########################################################################

def call_openai_chat(prompt: str, model: str, temperature: float = 0.2, n: int = 1, api_key: str | None = None) -> list[str]:
    """
    Volání OpenAI s podporou obou verzí SDK:
    - v1+: from openai import OpenAI; client.chat.completions.create(...)
    - legacy (<=0.28): openai.ChatCompletion.create(...)
    """
    # Pokus o nové SDK (v1+)
    try:
        from openai import OpenAI  # openai>=1.0
        if not api_key:
            return [f"[Missing API key] {prompt}"]
        client = OpenAI(api_key=api_key)
        outputs: list[str] = []
        for _ in range(n):
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an unbiased expert consultant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=800,
            )
            outputs.append(resp.choices[0].message.content or "")
        return outputs
    except Exception as v1_err:
        # Fallback na legacy SDK (<=0.28)
        try:
            import openai  # type: ignore
            if not api_key:
                return [f"[Missing API key] {prompt}"]
            openai.api_key = api_key
            outputs: list[str] = []
            for _ in range(n):
                resp = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an unbiased expert consultant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=800,
                )
                outputs.append(resp["choices"][0]["message"]["content"])
            return outputs
        except Exception as legacy_err:
            return [f"[OpenAI error] v1={v1_err} | legacy={legacy_err}"]


def run_analysis(selected_ids: list[str], provider: str, model: str, n_samples: int, api_key: str | None = None) -> pd.DataFrame:
    """
    Iterace přes vybrané scénáře, volání LLM a sběr odpovědí do DataFrame.
    Provider 'openai' používá call_openai_chat, jinak vrací mock odpovědi.
    """
    rows = []
    for sc in SCENARIOS:
        if sc["id"] not in selected_ids:
            continue
        for lang_key in ["cs", "sk", "en"]:
            query = sc[lang_key]
            if provider.lower() == "openai":
                responses = call_openai_chat(query, model=model, n=n_samples, api_key=api_key)
            else:
                responses = [f"[Mock response] {query}" for _ in range(n_samples)]
            for i, resp in enumerate(responses):
                rows.append(
                    {
                        "scenario_id": sc["id"],
                        "language": lang_key,
                        "query": query,
                        "assistant": provider,
                        "model": model,
                        "sample_ix": i + 1,
                        "raw_text": resp,
                    }
                )
    return pd.DataFrame(rows)


def export_dataframe_to_docx(df: pd.DataFrame) -> Path:
    """
    Export DataFrame do validního DOCX bez externích knihoven.
    Vytvoříme minimální OOXML balíček:
      - [Content_Types].xml
      - _rels/.rels       (odkaz na /word/document.xml)
      - word/document.xml (obsah)
    """
    import zipfile, html, tempfile
    from pathlib import Path

    tmp_dir = tempfile.mkdtemp()
    docx_path = Path(tmp_dir) / "results.docx"

    # --- obsah dokumentu: jednoduché odstavce (header + řádky tabulky) ---
    def esc(s: str) -> str:
        return html.escape(s if s is not None else "", quote=True)

    lines = []
    headers = list(df.columns)
    lines.append(" | ".join([str(h) for h in headers]))
    lines.append("-" * 8)
    for _, row in df.iterrows():
        vals = ["" if pd.isna(v) else str(v) for v in row.to_list()]
        lines.append(" | ".join(vals))

    text_lines = [esc("AI Visibility Monitor – Results")] + [esc(l) for l in lines]

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        + "".join(f"<w:p><w:r><w:t>{line}</w:t></w:r></w:p>" for line in text_lines)
        + '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/></w:sectPr>'
        "</w:body>"
        "</w:document>"
    )

    # hlavní RELS – musí ukazovat na word/document.xml
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        "</Relationships>"
    )

    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )

    with zipfile.ZipFile(docx_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types_xml)
        z.writestr("_rels/.rels", rels_xml)
        z.writestr("word/document.xml", document_xml)

    return docx_path


def get_download_link(file_path: Path, filename: str) -> str:
    """
    Vytvoří <a> odkaz pro stažení souboru (data URL, base64).
    """
    data = file_path.read_bytes()
    mime_map = {
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".md": "text/markdown",
        ".txt": "text/plain",
    }
    ext = file_path.suffix.lower()
    mime = mime_map.get(ext, "application/octet-stream")
    b64 = base64.b64encode(data).decode()
    label = ext.upper().lstrip(".")
    return f'<a href="data:{mime};base64,{b64}" download="{filename}">Download {label} file</a>'


##########################################################################
# Streamlit UI
##########################################################################

def main() -> None:
    st.set_page_config(page_title="AI Visibility Monitor (Seyfor)")
    st.title("AI Visibility Monitor – Seyfor Scenarios")
    st.write(
        "Vyberte scénáře a spusťte analýzu. Aplikace podporuje volání modelů "
        "přes knihovnu OpenAI nebo může vracet mock odpovědi. Po skončení lze "
        "výsledky stáhnout jako DOCX dokument."
    )

    # konfigurace
    st.sidebar.header("Configuration")
    provider = st.sidebar.selectbox("Provider", ["openai", "mock"], index=1)
    model = st.sidebar.text_input("Model name", value="gpt-4o-mini")
    api_key = st.sidebar.text_input("API key (OpenAI)", type="password")
    n_samples = st.sidebar.slider("Samples per query", min_value=1, max_value=5, value=1)

    # výběr scénářů
    st.subheader("Select scenarios")
    scenario_labels = [f"{sc['id']} – {sc['context']}" for sc in SCENARIOS]
    selected_labels = st.multiselect(
        "Vyberte scénáře", options=scenario_labels, default=scenario_labels[:3]
    )
    selected_ids = [SCENARIOS[i]["id"] for i, lbl in enumerate(scenario_labels) if lbl in selected_labels]

    if st.button("Run analysis"):
        if not selected_ids:
            st.warning("Please select at least one scenario.")
        else:
            with st.spinner("Running analysis..."):
                df_results = run_analysis(selected_ids, provider, model, n_samples, api_key)
            st.success("Analysis finished")
            st.subheader("Results")
            st.dataframe(df_results)

            # Export do DOCX
            st.subheader("Export")
            docx_path = export_dataframe_to_docx(df_results)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"aivm_results_{timestamp}.docx"
            st.markdown(get_download_link(docx_path, filename), unsafe_allow_html=True)

    # náhled scénářů
    with st.expander("Show scenario details"):
        for sc in SCENARIOS:
            st.markdown(f"**{sc['id']}** – {sc['context']}")
            st.write(f"Persona: {sc['persona']}")
            st.write(f"CS: {sc['cs']}")
            st.write(f"SK: {sc['sk']}")
            st.write(f"EN: {sc
