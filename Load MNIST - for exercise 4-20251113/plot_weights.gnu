set terminal pngcairo size 400,400
set output "poids_digit0.png"
set title "Poids projetés - digit 0"
unset key
set view map
set pm3d map
set palette rgbformulae 22,13,-31
splot 'poids_digit0.dat' matrix with image

