import ipyreact
import ipywidgets
import traitlets as t

css = ipywidgets.HTML(r"""
                      <style>
.token-container {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-auto-rows: auto;
    gap: 0.5ex 0.2rem;
    width: fit-content;
    line-height: 1.3rem;
    min-height: 3rem;
}

.token-container .encoded {
     font-size: 90%;
     font-weight: 100;
}

.token-container > div {
    display: flex;
    min-width: 3ex;
    justify-content: center;
    padding: 0 0.4rem;
    border-radius:0.4rem;
    flex-direction: column;
    align-items: center;
}</style>
""")


@t.signature_has_traits
class TokenMatrix(ipyreact.Widget):
    encoded = t.List(t.Int, sync=True)
    decoded = t.List(t.Unicode, sync=True)
    rows = t.List(t.Int, sync=True)
    cols = t.List(t.Int, sync=True)
    colors = t.List(t.Unicode, sync=True)
    classes = t.List(t.Unicode, sync=True)

    _esm = r"""
import * as React from "react";

export default function({ encoded, decoded, rows, cols, colors, classes }) {
  const combinedArray = decoded.map((value, i) => ({
    decoded: value,
    encoded: encoded[i],
    row: rows[i],
    col: cols[i],
    color: colors[i],
    className: classes[i]
  }));

  return (
    <div className="token-container">
      {combinedArray.map(({decoded, encoded, row, col, className, color}, index) => (
        <div
            data-encoded={encoded} data-decoded={decoded} style={{gridColumn:col, gridRow:row, backgroundColor:color}} className={className}>
                <span className="decoded">{decoded}</span>
                <span className="encoded">{encoded}</span>
        </div>
      ))}
    </div>
  );
}
"""
