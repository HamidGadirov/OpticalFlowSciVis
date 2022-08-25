#!/usr/bin/env python
from __future__ import print_function

import py_amira_file_reader.read_amira as read_amira
import numpy as np
import sys

import vtk
import argparse

def show_file(fname,gpu=False):
    data = read_amira.read_amira( fname )
    dlist = data['data']
    merged = {}
    for row in dlist:
        merged.update(row)
    if 'data' not in merged:
        print('Only binary .am files are supported',file=sys.stderr)
        sys.exit(1)
    arr = merged['data']

    dictRGB = {}
    if 'Materials' in merged['Parameters']:
        tdict = merged['Parameters']['Materials']
        for name in tdict:
            namedict=tdict[name]
            this_id = namedict['Id']
            dictRGB[this_id] = namedict.get('Color',[1.0,1.0,1.0])

    if 1:
        # from http://www.vtk.org/Wiki/VTK/Examples/Python/vtkWithNumpy

        # For VTK to be able to use the data, it must be stored as a VTK-image. This can be done by the vtkImageImport-class which
        # imports raw data and stores it.
        dataImporter = vtk.vtkImageImport()
        # The preaviusly created array is converted to a string of chars and imported.
        data_string = arr.tostring()
        dataImporter.CopyImportVoidPointer(data_string, len(data_string))
        # The type of the newly imported data is set to unsigned char (uint8)
        dataImporter.SetDataScalarTypeToUnsignedChar()
        # Because the data that is imported only contains an intensity value (it isnt RGB-coded or someting similar), the importer
        # must be told this is the case.
        dataImporter.SetNumberOfScalarComponents(1)

        dataImporter.SetDataExtent (0, arr.shape[0]-1, 0, arr.shape[1]-1, 0, arr.shape[2]-1)
        dataImporter.SetWholeExtent(0, arr.shape[0]-1, 0, arr.shape[1]-1, 0, arr.shape[2]-1)

    if 1:
        # from https://pyscience.wordpress.com/2014/11/16/volume-rendering-with-python-and-vtk/

        # Opacity of the different volumes (between 0.0 and 1.0)
        volOpacityDef = 0.25

        funcColor = vtk.vtkColorTransferFunction()
        if len(dictRGB):
            for idx in dictRGB.keys():
                funcColor.AddRGBPoint(idx,
                                      dictRGB[idx][0],
                                      dictRGB[idx][1],
                                      dictRGB[idx][2])
        else:
            maxval = float(np.max( arr ))
            print('Automatically scaling to maximum value %f'%maxval)
            funcColor.AddRGBPoint(  0.0, 0.0, 0.0, 0.0)
            funcColor.AddRGBPoint( maxval, 1.0, 1.0, 1.0)

        if 1:
            funcOpacityScalar = vtk.vtkPiecewiseFunction()

            for idx in dictRGB.keys():
                funcOpacityScalar.AddPoint(idx, volOpacityDef if idx!=0 else 0.0)


        if 1:
            funcOpacityGradient = vtk.vtkPiecewiseFunction()

            funcOpacityGradient.AddPoint(1,   0.0)
            funcOpacityGradient.AddPoint(5,   0.1)
            funcOpacityGradient.AddPoint(100,   1.0)

        if 1:
            propVolume = vtk.vtkVolumeProperty()
            propVolume.ShadeOff()
            propVolume.SetColor(funcColor)
            propVolume.SetScalarOpacity(funcOpacityScalar)
            propVolume.SetGradientOpacity(funcOpacityGradient)
            propVolume.SetInterpolationTypeToLinear()

        if 1:
            if not gpu:
                funcRayCast = vtk.vtkVolumeRayCastCompositeFunction()
                funcRayCast.SetCompositeMethodToClassifyFirst()

                mapperVolume = vtk.vtkVolumeRayCastMapper()
                mapperVolume.SetVolumeRayCastFunction(funcRayCast)
            else:
                mapperVolume = vtk.vtkGPUVolumeRayCastMapper()
                mapperVolume.SetBlendModeToMaximumIntensity();
                mapperVolume.SetSampleDistance(0.1)
                mapperVolume.SetAutoAdjustSampleDistances(0)

            mapperVolume.SetInputConnection(dataImporter.GetOutputPort())

            actorVolume = vtk.vtkVolume()
            actorVolume.SetMapper(mapperVolume)
            actorVolume.SetProperty(propVolume)

    if 1:
        # most from http://www.vtk.org/Wiki/VTK/Examples/Python/vtkWithNumpy

        renderer = vtk.vtkRenderer()
        renderWin = vtk.vtkRenderWindow()
        renderWin.AddRenderer(renderer)

        renderInteractor = vtk.vtkRenderWindowInteractor()
        renderInteractor.SetRenderWindow(renderWin)

        renderer.AddActor(actorVolume)
        renderer.SetBackground(0,0,0)
        renderWin.SetSize(800, 600)

        renderInteractor.Initialize()
        # Because nothing will be rendered without any input, we order the first render manually before control is handed over to the main-loop.
        renderWin.Render()
        renderInteractor.Start()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', type=str, help='The file to show')
    parser.add_argument(
        '--gpu', action='store_true', default=False,
        help='use GPU rendering',)
    args = parser.parse_args()
    show_file(args.FILE,gpu=args.gpu)

if __name__=='__main__':
    main()
