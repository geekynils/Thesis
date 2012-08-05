unset arrow
unset mouse
set title 'Mesh' font 'Arial,12'
set style line 1 pointtype 7 linecolor rgb 'gray'
set style line 2 pointtype 7 linecolor rgb 'black'
set style line 3 pointtype 7 linecolor rgb 'red'
set pointsize 2
set arrow from 2,0.75 to 1.95,0.55 ls 1
set arrow from 3.375,1.5 to 3.575,1.425 ls 1
set arrow from 2.625,3 to 2.725,3.225 ls 1
set arrow from 1.25,2.25 to 1,2.3 ls 1
set arrow from 4.25,0.625 to 4.275,0.375 ls 1
set arrow from 4.625,1.625 to 4.8,1.8 ls 1
set arrow from 4.125,3.125 to 4.25,3.05 ls 1
set arrow from 3.5,3.875 to 3.525,4.075 ls 1
set arrow from 2,3.75 to 1.95,3.85 ls 1
set arrow from 2,2 to 3.4,1 ls 1
set arrow from 3.4,1 to 4.25,1.5 ls 1
set arrow from 4.25,1.5 to 3.4,3.1 ls 1
plot '-' ls 2 with lines notitle
1 1
3 0.5
3 0.5
3.75 2.5
3.75 2.5
1.5 3.5
1.5 3.5
1 1
3 0.5
5.5 0.75
5.5 0.75
3.75 2.5
3.75 2.5
4.5 3.75
4.5 3.75
2.5 4
2.5 4
1.5 3.5
e
pause -1 'Hit OK to move to the next state'
set title 'Particle in cell 0' font 'Arial,12'
plot '-' ls 1 with lines notitle, '-' ls 3 with lines notitle
1 1
3 0.5
3 0.5
3.75 2.5
3.75 2.5
1.5 3.5
1.5 3.5
1 1
3 0.5
5.5 0.75
5.5 0.75
3.75 2.5
3.75 2.5
4.5 3.75
4.5 3.75
2.5 4
2.5 4
1.5 3.5
e
2 2
3.23239 1.11972
e
pause -1 'Hit OK to move to the next state'
set title 'Particle in cell 1' font 'Arial,12'
plot '-' ls 1 with lines notitle, '-' ls 3 with lines notitle
1 1
3 0.5
3 0.5
3.75 2.5
3.75 2.5
1.5 3.5
1.5 3.5
1 1
3 0.5
5.5 0.75
5.5 0.75
3.75 2.5
3.75 2.5
4.5 3.75
4.5 3.75
2.5 4
2.5 4
1.5 3.5
e
3.23239 1.11972
3.4 1
e
pause -1 'Hit OK to move to the next state'
set title 'Particle in cell 1' font 'Arial,12'
plot '-' ls 1 with lines notitle, '-' ls 3 with lines notitle
1 1
3 0.5
3 0.5
3.75 2.5
3.75 2.5
1.5 3.5
1.5 3.5
1 1
3 0.5
5.5 0.75
5.5 0.75
3.75 2.5
3.75 2.5
4.5 3.75
4.5 3.75
2.5 4
2.5 4
1.5 3.5
e
3.4 1
4.25 1.5
e
pause -1 'Hit OK to move to the next state'
set title 'Particle in cell 1' font 'Arial,12'
plot '-' ls 1 with lines notitle, '-' ls 3 with lines notitle
1 1
3 0.5
3 0.5
3.75 2.5
3.75 2.5
1.5 3.5
1.5 3.5
1 1
3 0.5
5.5 0.75
5.5 0.75
3.75 2.5
3.75 2.5
4.5 3.75
4.5 3.75
2.5 4
2.5 4
1.5 3.5
e
4.25 1.5
3.73707 2.46552
e
pause -1 'Hit OK to move to the next state'
set title 'Particle in cell 0' font 'Arial,12'
plot '-' ls 1 with lines notitle, '-' ls 3 with lines notitle
1 1
3 0.5
3 0.5
3.75 2.5
3.75 2.5
1.5 3.5
1.5 3.5
1 1
3 0.5
5.5 0.75
5.5 0.75
3.75 2.5
3.75 2.5
4.5 3.75
4.5 3.75
2.5 4
2.5 4
1.5 3.5
e
3.73707 2.46552
3.70909 2.51818
e
pause -1 'Hit OK to move to the next state'
set title 'Particle in cell 2' font 'Arial,12'
plot '-' ls 1 with lines notitle, '-' ls 3 with lines notitle
1 1
3 0.5
3 0.5
3.75 2.5
3.75 2.5
1.5 3.5
1.5 3.5
1 1
3 0.5
5.5 0.75
5.5 0.75
3.75 2.5
3.75 2.5
4.5 3.75
4.5 3.75
2.5 4
2.5 4
1.5 3.5
e
3.70909 2.51818
3.4 3.1
e
