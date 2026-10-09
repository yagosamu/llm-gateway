"""Write a self-contained HTML page for grading the blind human sheet in a browser, no spreadsheet needed.

The page shows the 30 items of sheet.csv one under the other with the rubric, a yes/no choice and a
note per item, keeps answers in the browser's local storage while grading, and downloads answers.csv
(item, acceptable, note) when done. The texts are embedded as JSON and rendered with textContent, so
nothing in a request or answer is interpreted as HTML.

Usage: uv run python -m harness.routing.human_page"""
import csv
import json
import sys

from harness.routing import prereg
from harness.routing.human_sheet import HUMAN_DIR

TEMPLATE = """<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Human grading</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:900px;margin:0 auto;padding:16px;line-height:1.5;background:#fafafa;color:#222}
 .item{background:#fff;border:1px solid #ddd;border-radius:8px;padding:16px;margin:20px 0}
 .item.done{border-color:#7a7}
 h2{margin:0 0 8px;font-size:1.1em}
 .label{font-weight:600;margin-top:12px}
 .text{white-space:pre-wrap;background:#f4f4f4;padding:8px;border-radius:4px;font-size:.95em}
 .choice label{margin-right:24px;font-size:1.05em}
 textarea{width:100%;min-height:40px;margin-top:6px}
 #bar{position:sticky;top:0;background:#fafafa;padding:8px 0;border-bottom:1px solid #ddd}
 button{font-size:1em;padding:6px 14px}
 details pre{white-space:pre-wrap}
</style></head><body>
<div id="bar"><span id="progress"></span> <button id="download">Baixar answers.csv</button></div>
<details><summary>Rubrica (a mesma que os juízes recebem)</summary><pre id="rubric"></pre></details>
<div id="items"></div>
<script>
const ITEMS = __ITEMS__;
const RUBRIC = __RUBRIC__;
const KEY = "llm-gateway-human-grading-v1";
let answers = {};
try { answers = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { answers = {}; }
function save() { try { localStorage.setItem(KEY, JSON.stringify(answers)); } catch (e) {} render(); }
document.getElementById("rubric").textContent = RUBRIC;
const root = document.getElementById("items");
function block(label, text) {
  const l = document.createElement("div"); l.className = "label"; l.textContent = label;
  const t = document.createElement("div"); t.className = "text"; t.textContent = text;
  return [l, t];
}
for (const it of ITEMS) {
  const box = document.createElement("div"); box.className = "item"; box.id = "item-" + it.item;
  const h = document.createElement("h2"); h.textContent = "Item " + it.item + " de " + ITEMS.length;
  box.append(h, ...block("Pedido", it.request), ...block("Resposta de referência", it.reference_answer),
             ...block("Resposta candidata", it.candidate_answer));
  const choice = document.createElement("div"); choice.className = "choice label";
  choice.append("Aceitável?  ");
  for (const v of ["yes", "no"]) {
    const lab = document.createElement("label"); const r = document.createElement("input");
    r.type = "radio"; r.name = "a" + it.item; r.value = v;
    r.checked = (answers[it.item] || {}).acceptable === v;
    r.onchange = () => { answers[it.item] = {...(answers[it.item] || {}), acceptable: v}; save(); };
    lab.append(r, " " + (v === "yes" ? "sim (yes)" : "não (no)")); choice.append(lab);
  }
  const note = document.createElement("textarea"); note.placeholder = "Nota opcional (ajuda nos casos de não)";
  note.value = (answers[it.item] || {}).note || "";
  note.oninput = () => { answers[it.item] = {...(answers[it.item] || {}), note: note.value}; save(); };
  box.append(choice, note); root.append(box);
}
function render() {
  let n = 0;
  for (const it of ITEMS) {
    const ok = (answers[it.item] || {}).acceptable;
    document.getElementById("item-" + it.item).classList.toggle("done", !!ok); if (ok) n++;
  }
  document.getElementById("progress").textContent = n + " de " + ITEMS.length + " respondidos";
}
function csvCell(s) { return '"' + String(s || "").replace(/"/g, '""') + '"'; }
document.getElementById("download").onclick = () => {
  const missing = ITEMS.filter(it => !(answers[it.item] || {}).acceptable).map(it => it.item);
  if (missing.length && !confirm("Faltam os itens " + missing.join(", ") + ". Baixar mesmo assim?")) return;
  const lines = ["item,acceptable,note"].concat(ITEMS.map(it => {
    const a = answers[it.item] || {}; return [it.item, a.acceptable || "", csvCell(a.note)].join(",");
  }));
  const blob = new Blob([lines.join("\\n") + "\\n"], {type: "text/csv;charset=utf-8"});
  const link = document.createElement("a"); link.href = URL.createObjectURL(blob);
  link.download = "answers.csv"; link.click();
};
render();
</script></body></html>
"""


def build_page(items: list[dict]) -> str:
    # json.dumps escapes nothing HTML-specific; replacing "</" keeps a text containing "</script>"
    # from closing the script element early.
    as_js = lambda value: json.dumps(value, ensure_ascii=False).replace("</", "<\\/")
    return TEMPLATE.replace("__ITEMS__", as_js(items)).replace("__RUBRIC__", as_js(prereg.RUBRIC))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    with (HUMAN_DIR / "sheet.csv").open(encoding="utf-8-sig", newline="") as f:
        items = [{k: row[k] for k in ("item", "request", "reference_answer", "candidate_answer")}
                 for row in csv.DictReader(f)]
    path = HUMAN_DIR / "grade.html"
    path.write_text(build_page(items), encoding="utf-8", newline="\n")
    print(f"{len(items)} items -> {path}")


if __name__ == "__main__":
    main()
