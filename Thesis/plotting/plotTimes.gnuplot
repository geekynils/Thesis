set title "CPU vs GPU"

set xlabel "Number of Particles (* 10'000)"
set ylabel "Time in Microseconds"

plot "cpuTimes.txt" u ($0+1):1 every 1::::50 t "CPU" w lp, \
     "gpuTimes.txt" u ($0+1):1 every 1::::50 t "GPU" w lp