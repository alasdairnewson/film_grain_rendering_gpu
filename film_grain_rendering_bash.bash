#!/bin/bash


mIn=384
nIn=512
xCentre=$(echo $nIn/2 | bc -l)
yCentre=$(echo $mIn/2 | bc -l)
nLevels=1
r=0.03
fileNameIn="input/voiture_compressed.png"
fileNameOut="results/voiture_out"
#zoom info
currZoom=1.0
zoomCoef=0.90
#carry out grain rendering, first level
fileNameOutTemp=($fileNameOut"_level_"$(printf "%03d" 1)".png")
./bin/film_grain_rendering_main $fileNameIn $fileNameOutTemp -r $r -sigmaR 0.0 -zoom 1 -color 0 -NmonteCarlo 800

for i in `seq 2 $nLevels`;
do
	currZoom=$(echo $currZoom/$zoomCoef | bc -l)
	currZoomTemp=$(echo $currZoom*2 | bc -l)
	nTemp=$(echo $nIn/$currZoomTemp | bc -l)
	mTemp=$(echo $mIn/$currZoomTemp | bc -l)
	xA=$(echo $xCentre-$nTemp | bc -l)
	xB=$(echo $xCentre+$nTemp | bc -l)
	yA=$(echo $yCentre-$mTemp | bc -l)
	yB=$(echo $yCentre+$mTemp | bc -l)
	fileNameOutTemp=($fileNameOut"_level_"$(printf "%03d" $i)".png")
	#carry out grain rendering
	./bin/film_grain_rendering_main input/fleur.png $fileNameOutTemp -r $r -color 0 -NmonteCarlo 500 -xA $xA -yA $yA -xB $xB -yB $yB -height $m -width $n
done

#./bin/film_grain_rendering_main input/fleur.png fleur_out.tiff -r 0.05 -color 1 -NmonteCarlo 800 -zoom 7

#r="0.05"
#nMonteCarloArray=("10" "20" "40" "80" "160" "320" "640" "1080")

#cary_grant_cigarette
#inputPathFileArray=( "/home/dist/alasdairnewson/Content/Wizard_oz_b.tiff" )
#outputFile="Wizard_oz_b_out_"
#outputFilePath="Results/"
#outputFile="Constant_out_"
#inputPathFileArray=( "/home/alasdair/Alasdair/Postdoc/2015_descartes/Content/Film_grain/Cary_grant_small.tiff" )
#inputPathFileArray=( "/home/dist/alasdairnewson/Content/Constant_image_128_size_02048.tiff" )
#outputFile="Cary_grant_small_out_"
#for i in "${inputPathFileArray[@]}"
#do
#echo $i
#    for j in "${nMonteCarloArray[@]}"
#	do
#		echo $j
#		./film_grain_synthesis_main $i $outputFile -r $r -grainSigma "0.0" -filterSigma "0.8" -zoom "1.0" -"NmonteCarlo" $j
#	done
#done
