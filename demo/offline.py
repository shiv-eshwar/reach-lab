"""Package the complete study replay into a single file for offline projection."""
from pathlib import Path

def export_replay():
    demo = Path(__file__).resolve().parent
    page = (demo / "index.html").read_text()
    page = page.replace('<link rel="stylesheet" href="/style.css">',
                        "<style>" + (demo/"style.css").read_text() + "</style>")
    page = page.replace('<button id="liveTab"', '<button hidden id="liveTab"')
    # Escape HTML delimiters in JSON before embedding it in a script element.
    data = (demo/"study.json").read_text().replace("<",r"\u003c")
    scripts = '<script id="studyData" type="application/json">'+data+"</script>"
    scripts += "<script>" + (demo/"app.js").read_text() + "</script>"
    page = page.replace('<script src="/app.js"></script>', scripts)
    output = demo.parent / "presentation" / "Offline replay.html"
    output.parent.mkdir(exist_ok=True)
    output.write_text(page, encoding="utf-8")

if __name__ == "__main__":
    export_replay()
