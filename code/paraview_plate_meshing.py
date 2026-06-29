import os
import sys

import paraview.simple


def main():
    assert len(sys.argv) >= 6
    folder_path = sys.argv[1]
    filename_image = os.path.join(folder_path, sys.argv[2])
    filename_mesh = os.path.join(folder_path, sys.argv[3])
    thickness = float(sys.argv[4])
    TargetReduction = float(sys.argv[5])
    scale_factor = None
    if len(sys.argv) > 6:
        scale_factor = float(sys.argv[6])
    print("*** Paraview: image to plate with thickness " + str(thickness))
    print("\t* Loading image data at " + filename_image)
    print("\t- [IMPORTANT]: Input image **MUST** have one pixel border with zeroes")
    image = paraview.simple.PNGSeriesReader(
        registrationName="image",
        FileNames=[filename_image],
    )
    image.UpdatePipeline()

    if scale_factor is None:
        # by default, normalize to dimension one for the longest coordinate dimension
        extent = image.GetDataInformation().GetExtent()
        size_0 = extent[1] - extent[0] - 1
        size_1 = extent[3] - extent[2] - 1
        size_norm = max(size_0, size_1)
        scale_factor = 1 / size_norm
    print("\t* Scale factor = " + str(scale_factor))
    transform = paraview.simple.Transform(image)
    transform.Transform.Scale = [scale_factor, scale_factor, 1]
    transform.UpdatePipeline()

    low_tresh = 0.5
    isosurfaces = [0.5, 255 / 2]
    print("\t* Countouring data at isosurfaces " + str(isosurfaces) + "...")
    contours = paraview.simple.Contour(
        registrationName="contours",
        Input=transform,
        Isosurfaces=isosurfaces,
        ComputeNormals=0,
        ComputeGradients=0,
        GenerateTriangles=0,
    )
    contours.UpdatePipeline()

    print("\t* Computing Delaunay triangulation of contours...")
    delaunay2d = paraview.simple.Delaunay2D(
        registrationName="delaunay2d",
        Input=contours,
        Tolerance=0.0,
    )
    delaunay2d.UpdatePipeline()

    print("\t* Thresholding triangulation")
    thresholded_delaunay2d = paraview.simple.Threshold(
        registrationName="thresholded_delaunay2d",
        Input=delaunay2d,
        UpperThreshold=low_tresh + 1,
        ThresholdMethod="Above Upper Threshold",
    )
    thresholded_delaunay2d.UpdatePipeline()

    print("\t* Extracting surface...")
    surface = paraview.simple.ExtractSurface(
        registrationName="surface",
        Input=thresholded_delaunay2d,
    )
    surface.UpdatePipeline()

    print("\t* Simplifying triangulation...")
    surface_decimated = paraview.simple.Decimate(
        registrationName="surface_decimated",
        Input=surface,
        PreserveTopology=1,
        TargetReduction=TargetReduction,
    )
    surface_decimated.UpdatePipeline()

    print("\t* Reorient triangles...")
    surface_reorient = paraview.simple.SurfaceNormals(
        registrationName="surface_reorient",
        Input=surface_decimated,
        NonManifoldTraversal=0,
        Splitting=0,
        PieceInvariant=1,
        Consistency=1,
    )

    print("\t* Linear extrusion in the Z direction...")
    extruded_mesh = paraview.simple.LinearExtrusion(surface_reorient)
    extruded_mesh.Vector = [0, 0, 1]
    extruded_mesh.ScaleFactor = thickness
    extruded_mesh.UpdatePipeline()

    print("\t* Triangulate side faces...")
    triangulated_mesh = paraview.simple.Triangulate(
        registrationName="triangulated_mesh",
        Input=extruded_mesh,
    )
    triangulated_mesh.UpdatePipeline()

    print("\t* Export to " + filename_mesh + "...")
    paraview.simple.SaveData(filename_mesh, proxy=triangulated_mesh, FileType="Binary")


if __name__ == "__main__":
    main()
