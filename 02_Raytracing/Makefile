all:
	g++ -std=c++17 main.cpp -O3 -mavx -DUSE_AVX -ftree-vectorize -fopt-info-vec-optimized -fopenmp -I../01_KickSimulator/ -o main -lm

# Default values for simulation parameters
METHOD ?= 0
DEPTH ?= 5

run_simulation: all
	@mkdir -p output
	./main $(METHOD) $(DEPTH)
	ffmpeg -y -framerate 30 -pattern_type glob -i 'output/frame_*.jpg' -c:v libx264 -pix_fmt yuv420p -q:v 2 free_kick_video.mp4

generate_video:
	ffmpeg -framerate 30 -pattern_type glob -i 'output/frame_*.jpg' -c:v libx264 -pix_fmt yuv420p -q:v 2 free_kick_animation.mp4

generate_videos_per_folder:
	@echo "Generating videos for each wavepoint folder..."
	@for dir in $(shell find ./output/ -maxdepth 1 -type d -name 'wavepoints_*'); do \\
		echo "Processing $$dir"; \\
		video_name=$$(basename $$dir); \\
		ffmpeg -y -framerate 30 -pattern_type glob -i "$$dir/frame_*.jpg" -c:v libx264 -pix_fmt yuv420p -q:v 2 "$$dir/$${video_name}.mp4"; \\
	done
	@echo "Finished generating videos."

free_kick_workflow: all
	$(MAKE) run_simulation METHOD=$(METHOD) DEPTH=$(DEPTH)
	$(MAKE) generate_video

fresnel: all
	@mkdir -p output
	./main 0 5
schlick: all
	@mkdir -p output
	./main 1 5
profiling:
	gprof ./main gmon.out > profile.txt
create_sphere_animation:
	ffmpeg -framerate 30 -pattern_type glob -i 'output/sphere/*.jpg' -q:v 2 sphere.gif
create_torus_animation:
	ffmpeg -framerate 30 -pattern_type glob -i 'output/torus/*.jpg' -q:v 2 torus.gif
benchmark: all
	@echo "param1,param2,time" > benchmark_results.csv
	@mkdir -p output
	@for i in 0 1; do \
	    for j in 1 2 3 4 5 6 7 8 9 10; do \
	        START=$$(date +%s%N); \
	        ./main $$i $$j > /dev/null; \
	        END=$$(date +%s%N); \
	        TIME=$$(echo "scale=6; ($$END - $$START) / 1000000000" | bc); \
	        echo "$$i,$$j,$$TIME" >> benchmark_results.csv; \
	    done \
	done
clean:
	rm -rf output/*
	rm -f main free_kick_animation.mp4 gmon.out profile.txt benchmark_results.csv
