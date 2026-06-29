filename="../results/isodither.json"
python pixels_per_point.py $filename
python mean_fluctuations.py $filename
python convergence.py $filename
python varying_poisson.py $filename
python explore.py $filename
python interpolation.py $filename
python fea_halftonings.py $filename
python teaser.py $filename
pvpython paraview_plate_meshing.py ../results/isodither/ teaser.png teaser.ply 0.02 0.7
python curve_interpolation_to_glsl.py $filename
