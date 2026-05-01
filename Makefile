.PHONY: all build slides clean

all: build slides

build:
	python3 src/build.py

slides:
	python3 src/compile.py

clean:
	rm -rf output slides_letemps_parlacap.pdf slides/*.aux slides/*.log slides/*.nav slides/*.out slides/*.snm slides/*.toc slides/*.fdb_latexmk slides/*.fls
