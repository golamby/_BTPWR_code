"""Rebuild the private panels on the original 1800 by 1560 vector artwork."""
import argparse
import io
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
from matplotlib.figure import Figure

SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
ET.register_namespace("dc", "http://purl.org/dc/elements/1.1/")
ET.register_namespace("cc", "http://creativecommons.org/ns#")
ET.register_namespace("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
matplotlib.use("Agg")
matplotlib.rcParams.update({"mathtext.fontset": "stix", "svg.fonttype": "path"})
NAVY, TEAL, INK, GOLD, VIOLET = "#244B6B", "#2E6258", "#202B35", "#8E7340", "#624777"


def rebuild(source, destination):
    root = ET.parse(source).getroot()
    old = list(root)
    assert root.get("viewBox") == "0 0 1800 1560"
    assert old[177].text == "Signed RR packet + weight commitment"
    assert old[329].text.startswith("Authorized requester")
    for i in (178, 204, 308, 312, 314, 321, 322, 330):
        root.remove(old[i])
    replacements = {
        177: "Signed RR offer for the frozen round",
        240: "Freeze context",
        243: "3-of-4 match",
        246: "3-of-4 update",
        277: "Locked worker roster",
        309: "Signed readings for the authorized task",
        315: "Same admitted reports",
        317: "Quorum lock",
        320: "Settle state",
        323: "Freeze → Lock → Collect → Settle",
        324: "Bind phase, round and previous root",
        329: "Private execution · four-party secure computation",
        333: "Frozen context",
        334: "Quorum votes",
    }
    for i, content in replacements.items():
        old[i].text = content
    old[315].set("x", "440")
    old[315].set("font-size", "23")
    old[329].set("font-size", "26")
    old[331].set("d", "M772,1073 V1144")
    old[332].set("d", "M982,1144 V1073")
    old[333].set("x", "786")
    old[333].set("font-size", "19")
    old[334].set("x", "996")
    old[334].set("font-size", "19")

    def node(tag, parent=None, **attributes):
        return ET.SubElement(root if parent is None else parent, "{" + SVG + "}" + tag,
                             {k.replace("_", "-"): str(v) for k, v in attributes.items()})

    def text(x, y, label, size=24, color=INK, bold=False, center=False):
        e = node("text", x=x, y=y, font_size=size, fill=color,
                 font_weight="bold" if bold else "normal",
                 text_anchor="middle" if center else "start")
        e.text = label

    def box(x, y, w, h, fill, stroke=TEAL, radius=8, width=1.4):
        node("rect", x=x, y=y, width=w, height=h, rx=radius, fill=fill,
             stroke=stroke, stroke_width=width)

    def path(d, color=TEAL, arrow=False, dashed=False):
        attrs = {"d": d, "fill": "none", "stroke": color, "stroke-width": "2.1",
                 "stroke-linejoin": "round", "stroke-linecap": "round"}
        if arrow:
            attrs["marker-end"] = f"url(#{color.lstrip('#')})"
        if dashed:
            attrs["stroke-dasharray"] = "6 4"
        ET.SubElement(root, "{" + SVG + "}path", attrs)

    serial = 0

    def eq(expr, x, y, max_width=600, size=24, color=INK):
        nonlocal serial
        serial += 1
        f = Figure(figsize=(8, .40))
        f.text(0, 0, "$" + expr + "$", fontsize=12, color=color)
        output = io.BytesIO()
        f.savefig(output, format="svg", bbox_inches="tight", pad_inches=.012, transparent=True)
        fragment = ET.fromstring(output.getvalue())
        width = float(fragment.get("width").replace("pt", ""))
        scale = min(size / 12, max_width / width)
        prefix = f"committee_eq{serial}_"
        for item in fragment.iter():
            if item.get("id"):
                item.set("id", prefix + item.get("id"))
            for key, value in list(item.attrib.items()):
                if key.endswith("href") and value.startswith("#"):
                    item.set(key, "#" + prefix + value[1:])
                elif "url(#" in value:
                    item.set(key, value.replace("url(#", "url(#" + prefix))
        group = node("g", transform=f"translate({x},{y}) scale({scale})")
        for item in list(fragment):
            if not item.tag.endswith("metadata"):
                group.append(item)

    def step(number, x, y, label):
        node("circle", cx=x, cy=y, r=22, stroke=TEAL, stroke_width=2,
             fill="white")
        text(x, y+9, str(number), 29, TEAL, True, True)
        text(x+39, y+9, label, 29, TEAL, True)

    def matrix(x, y, values, color, pitch=25):
        for j, row in enumerate(values):
            for i, filled in enumerate(row):
                box(x+i*pitch, y+j*pitch, pitch-3, pitch-3, color if filled else "#F4F8F6",
                    "#82A79B", 2, .8)

    def server(x, y, number, color):
        node("ellipse", cx=x+39, cy=y+54, rx=44, ry=10, fill="#C5D9D3")
        node("polygon", points=f"{x},{y+11} {x+18},{y} {x+84},{y} {x+68},{y+11}",
             fill="#D3E5E2", stroke=color, stroke_width=1.5)
        node("polygon", points=f"{x+68},{y+11} {x+84},{y} {x+84},{y+43} {x+68},{y+53}",
             fill="#719891", stroke=color, stroke_width=1.5)
        box(x, y+11, 68, 42, "#FBFEFD", color, 2, 1.7)
        for offset in (19, 30, 41):
            path(f"M{x+12},{y+offset} h31", color)
            node("circle", cx=x+54, cy=y+offset, r=2.6, fill=color)
        eq(r"\mathcal{A}_" + str(number), x+25, y+68, 65, 24, color)

    def credential(x, y, label, color):
        node("circle", cx=x, cy=y, r=18, fill="#F8FBFA", stroke=color, stroke_width=1.6)
        node("circle", cx=x, cy=y-5, r=3.8, fill=color)
        path(f"M{x-7},{y+6} q7,-12 14,0", color)
        text(x, y+42, label, 20, color, True, True)

    def lock(x, y, color, fill):
        path(f"M{x-7},{y} v-7 a7,7 0 0 1 14,0 v7", color)
        box(x-11, y, 22, 16, fill, color, 2, 1.3)
        node("circle", cx=x, cy=y+6, r=2, fill=color)
        path(f"M{x},{y+7} v4", color)

    eq(r"\sigma_i=\mathrm{Sign}_i(t,\widetilde{\mathbf{b}}_{i,t},\widetilde{\mathbf{c}}_{i,t})",
       1040, 581, 532, 23, NAVY)
    eq(r"(t,I_t,J_t,B_0)", 1192, 770, 383, 24)
    eq(r"m_{hi,t}=(t,h,\alpha_i,x_{hi,t})", 142, 920, 440, 25, NAVY)
    eq(r"\sigma_{hi,t}=\mathrm{Sign}_i(H(m_{hi,t}))", 58, 990, 365, 21, NAVY)
    eq(r"J_t\longrightarrow J_{t+1}", 1492, 1025, 248, 25)
    eq(r"Q_{\mathrm{lock}},\quad Q_{\mathrm{settle}}", 1010, 971, 326, 24, VIOLET)

    step(3, 51, 1165, "Progress-aware recruitment")
    step(5, 1035, 1165, "Private feedback and recovery")
    path("M620,1196 V1482 M1002,1196 V1482", "#A1BFB3", dashed=True)
    path("M1372,1200 V1482", "#A1BFB3")

    text(52, 1220, "Forecast support", 24, TEAL, True)
    matrix(55, 1235, [[1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 1, 0]], "#4C8272", 23)
    eq(r"\pi_{ij,t},\ E_{ih,t}", 148, 1255, 112, 24)
    path("M262,1268 H306", TEAL, True)
    text(322, 1220, "Qualification progress", 24, TEAL, True)
    for k, x in enumerate((345, 416, 487, 558)):
        node("circle", cx=x, cy=1268, r=22, fill="#F4F8F6", stroke=TEAL,
             stroke_width=1.6)
        eq((r"U_" + str(3-k)) if k < 3 else "C", x-13, 1255, 28, 23, TEAL)
        if k < 3:
            path(f"M{x+23},1268 h23", TEAL, True)
    text(452, 1314, "Earliest under ideal scores", 18.5, TEAL, False, True)
    path("M50,1322 H592", "#A1BFB3")
    lock(62, 1343, TEAL, "#EAF3EF")
    text(84, 1359, "Weight mapping and exact payment", 24, TEAL, True)
    eq(r"\widehat r_{i,t}\ \mapsto\ w_{i,t},\quad \widehat c_t(S)\leq B_0",
       49, 1371, 537, 27)
    eq(r"j^\star=\min\,\arg\max_{j\in\{0,1,2\}}\widehat v_t(S_t^j)",
       49, 1424, 537, 25)
    text(49, 1498, "Mapped  /  C-seeded  /  rollout-greedy", 21, TEAL)

    text(810, 1174, "Four computing parties", 24, TEAL, True, True)
    path("M732,1248 H875 M732,1370 H875 M690,1293 V1323 M915,1293 V1323",
         "#7B9A91", dashed=True)
    path("M728,1252 L875,1366 M875,1252 L728,1366", "#7B9A91", dashed=True)
    node("circle", cx=803, cy=1308, r=31, fill="#EAF3EF", stroke=TEAL, stroke_width=2)
    lock(803, 1304, TEAL, "#D1E3DB")
    server(648, 1219, 1, TEAL)
    server(875, 1219, 2, TEAL)
    server(648, 1343, 3, TEAL)
    server(875, 1343, 4, TEAL)
    eq(r"\langle\bar r_{i,t}\rangle_1,\ldots,\langle\bar r_{i,t}\rangle_4",
       664, 1443, 292, 24, VIOLET)
    text(810, 1500, "Private ranks and masked updates", 21, TEAL, False, True)
    path("M596,876 H698 V1138 H650 V1192 H672", "#444B53", True)
    path("M642,876 V903", NAVY, True)
    credential(642, 929, "Requester R", NAVY)
    path("M591,1308 H643 M963,1308 H1017", TEAL, True)

    text(1026, 1218, "Other-credible feedback", 24, TEAL, True)
    for x, label, color in ((1061, "Peer", TEAL), (1138, "Peer", TEAL),
                            (1215, "Peer", TEAL), (1292, "Self", GOLD)):
        credential(x, 1254, label, color)
    path("M1080,1254 H1119 M1157,1254 H1196", "#7B9A91", dashed=True)
    path("M1277,1239 l30,30", GOLD)
    eq(r"\mathcal{I}_{hi,t}^{-i}=W^C_{h,t}\setminus\{i\}", 1027, 1313, 324, 23)
    eq(r"z_{hi,t}^{-i}=\mathrm{TD}(\mathcal{I}_{hi,t}^{-i})", 1029, 1364, 315, 23)
    eq(r"e_{hi,t}=|\check{x}_{hi,t}-z_{hi,t}^{-i}|/\Delta_T",
       1028, 1424, 315, 23)
    text(1026, 1505, "Residual iteration; no self-reference", 20, TEAL)

    text(1395, 1218, "Credible-source recovery", 24, TEAL, True)
    matrix(1398, 1235, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "#447D6D", 21)
    path("M1471,1265 H1513", TEAL, True)
    matrix(1523, 1235, [[1, 1, 0], [1, 1, 1], [0, 1, 1]], "#759A8F", 21)
    path("M1598,1265 H1638", TEAL, True)
    matrix(1648, 1235, [[1, 1, 1], [1, 1, 1], [0, 1, 1]], "#7796AC", 21)
    text(1427, 1321, "Anchors", 19, TEAL, False, True)
    text(1554, 1321, "Kernel", 19, TEAL, False, True)
    text(1679, 1321, "Estimates", 19, NAVY, False, True)
    eq(r"m_h(z^C_{\mathcal{A},t})=b_h+\alpha_{:h}^{\mathsf{T}}\mathbf{k}(z^C_{\mathcal{A},t})",
       1394, 1364, 353, 23)
    eq(r"\widehat{\mathbf{g}}_t\longrightarrow R,\quad \bar r_{i,t+1}+m_i\longrightarrow W_i",
       1395, 1434, 352, 23, VIOLET)
    text(1395, 1505, "Task estimates and masked worker update", 19.5, TEAL)
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    a = parser.parse_args()
    rebuild(a.source, a.output)
