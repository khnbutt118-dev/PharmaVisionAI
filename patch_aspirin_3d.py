from pathlib import Path
from datetime import datetime
import shutil
import subprocess
import sys
import re

root = Path.cwd()
gesture = root / "src/hand_gesture/air_canvas_live.py"
camera = root / "src/camera/pharmasight_live.py"

if not gesture.exists() or not camera.exists():
    raise SystemExit("ERROR: Run from PharmaSight-AI project root.")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_g = gesture.with_name(gesture.name + ".backup_" + stamp)
backup_c = camera.with_name(camera.name + ".backup_" + stamp)

shutil.copy2(gesture, backup_g)
shutil.copy2(camera, backup_c)

try:
    g = gesture.read_text(encoding="utf-8")
    c = camera.read_text(encoding="utf-8")

    # Locate the aspirin section precisely.
    start = g.find("ASPIRIN_ATOMS = [")
    end = g.find("# Capsule Explorer demonstration data.", start)

    if start < 0 or end < 0:
        raise RuntimeError("Aspirin section boundaries not found.")

    # Include the preceding aspirin explanatory comment.
    comment_start = g.rfind("# Aspirin", 0, start)
    if comment_start >= 0:
        start = comment_start

    replacement = '''# Aspirin: genuine RDKit-generated optimized 3D conformer
from rdkit import Chem
from rdkit.Chem import AllChem

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
ASPIRIN_MOL = Chem.AddHs(Chem.MolFromSmiles(ASPIRIN_SMILES))
_params = AllChem.ETKDGv3()
_params.randomSeed = 42

if AllChem.EmbedMolecule(ASPIRIN_MOL, _params) != 0:
    raise RuntimeError("RDKit aspirin 3D embedding failed")

AllChem.MMFFOptimizeMolecule(ASPIRIN_MOL)
_conf = ASPIRIN_MOL.GetConformer()

ASPIRIN_ATOMS = []
for _i, _atom in enumerate(ASPIRIN_MOL.GetAtoms()):
    _p = _conf.GetAtomPosition(_i)
    ASPIRIN_ATOMS.append((_atom.GetSymbol(), _p.x, _p.y, _p.z))

ASPIRIN_BONDS = [
    (b.GetBeginAtomIdx(), b.GetEndAtomIdx(),
     2 if b.GetBondType() == Chem.BondType.DOUBLE else 1)
    for b in ASPIRIN_MOL.GetBonds()
]

ATOM_COLORS = {
    "C": (185, 190, 200),
    "O": (55, 65, 245),
    "H": (235, 235, 235),
    "N": (245, 90, 60),
}

def draw_aspirin(frame, yaw, pitch, scale, center):
    cx, cy = int(center[0]), int(center[1])
    coords = np.array([a[1:] for a in ASPIRIN_ATOMS], dtype=float)
    coords *= 65.0 * scale

    cyaw, syaw = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    ry = np.array([[cyaw,0,syaw],[0,1,0],[-syaw,0,cyaw]])
    rx = np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]])
    rot = coords @ (ry @ rx).T

    projected, depths = [], []
    for x,y,z in rot:
        persp = 700.0 / max(250.0, 700.0 + z)
        projected.append((int(cx+x*persp), int(cy+y*persp)))
        depths.append(z)

    # Draw bonds first.
    for i,j,order in ASPIRIN_BONDS:
        p1,p2 = projected[i],projected[j]
        cv2.line(frame,p1,p2,(175,185,200),3,cv2.LINE_AA)
        if order == 2:
            dx,dy=p2[0]-p1[0],p2[1]-p1[1]
            length=max(1.0,math.hypot(dx,dy))
            ox,oy=int(-dy/length*3),int(dx/length*3)
            cv2.line(frame,(p1[0]+ox,p1[1]+oy),
                     (p2[0]+ox,p2[1]+oy),(210,220,235),1,cv2.LINE_AA)

    # Depth-sorted ball-and-stick atoms.
    for idx in sorted(range(len(ASPIRIN_ATOMS)),key=lambda k:depths[k]):
        element=ASPIRIN_ATOMS[idx][0]
        x,y=projected[idx]
        radius={"C":10,"O":11,"H":5,"N":11}.get(element,9)
        color=ATOM_COLORS.get(element,(180,180,180))
        cv2.circle(frame,(x,y),radius+2,(245,245,245),1,cv2.LINE_AA)
        cv2.circle(frame,(x-2,y-2),max(2,radius-2),
                   tuple(min(255,v+35) for v in color),-1,cv2.LINE_AA)
        cv2.circle(frame,(x,y),radius,color,-1,cv2.LINE_AA)
        cv2.circle(frame,(x-3,y-3),max(1,radius//3),
                   (235,235,235),-1,cv2.LINE_AA)

    cv2.putText(frame,"ASPIRIN | C9H8O4 | RDKit 3D",
                (cx-170,max(145,cy-205)),
                cv2.FONT_HERSHEY_SIMPLEX,.48,(120,255,225),1,cv2.LINE_AA)
    cv2.putText(frame,"INDEX: ROTATE | PINCH: ZOOM | TWO HANDS: TWIST/SCALE",
                (max(12,cx-300),min(frame.shape[0]-30,cy+205)),
                cv2.FONT_HERSHEY_SIMPLEX,.36,(220,230,245),1,cv2.LINE_AA)

'''

    g = g[:start] + replacement + g[end:]

    # Replace the exact observed rotation statements.
    old = '''self.yaw += dx * 0.004
                    self.pitch = float(np.clip(
                        self.pitch + dy * 0.003, -1.3, 1.3
                    ))'''

    new = '''self.yaw += dx * 0.008
                    self.pitch = float(np.clip(
                        self.pitch + dy * 0.006, -1.3, 1.3
                    ))'''

    if old not in c:
        raise RuntimeError("Exact rotation block not found; refusing camera edit.")

    c = c.replace(old, new, 1)

    gesture.write_text(g, encoding="utf-8")
    camera.write_text(c, encoding="utf-8")

    for f in (gesture, camera):
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(f)],
            capture_output=True, text=True
        )
        if result.returncode:
            raise RuntimeError(result.stderr)

    print("SUCCESS: RDKit 3D aspirin patch applied.")
    print("Backups:", backup_g.name, backup_c.name)
    print("Rotation sensitivity increased 2x.")
    print("Both Python files passed syntax checks.")

except Exception as e:
    shutil.copy2(backup_g, gesture)
    shutil.copy2(backup_c, camera)
    print("PATCH FAILED:", e)
    print("Original files restored.")
    raise SystemExit(1)
