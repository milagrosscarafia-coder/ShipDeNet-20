from IPython.display import Image, display
import glob

for f in sorted(glob.glob("/content/ShipDeNet-20/losses_*.png")):
    print(f)
    display(Image(f))

display(Image("/content/ShipDeNet-20/resultados/detecciones.png"))
display(Image("/content/ShipDeNet-20/resultados/curva_pr.png"))
!ls /content/ShipDeNet-20/*.png /content/ShipDeNet-20/resultados/
