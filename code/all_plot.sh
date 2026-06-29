filename="../results/isodither.json"
python geometry_plot.py $filename
python pixels_per_point_plot.py $filename
python mean_fluctuations_plot.py $filename
python convergence_plot.py $filename
python varying_poisson_plot.py $filename
python bulkshear_plot.py $filename
python betavarphi_plot.py $filename
python youngpoissoninterp_plot.py $filename
python constantyoung_plot.py $filename
python comparison_periodic_plot.py $filename
python halftoned_images_plot.py $filename
python fea_halftonings_plot.py $filename
python fea_boundary_conditions_plot.py $filename
python fea_halftonings_error_plot.py $filename
blender --background --python blender_plate_rendering.py -- $filename
python teaser_plot.py $filename
python pavilion_plot.py $filename
