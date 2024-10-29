# Fundamentals of power electronics course
[![Build LaTeX PDFs](https://github.com/IAS-Uni-Siegen/PE_course/actions/workflows/buildPDFs.yml/badge.svg?event=workflow_dispatch)](https://github.com/IAS-Uni-Siegen/PE_course/actions/workflows/buildPDFs.yml)
[![CC BY 4.0][cc-by-shield]][cc-by]
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-latex](https://img.shields.io/badge/Made%20with-LaTeX-1f425f.svg)](https://www.latex-project.org/)

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://licensebuttons.net/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

## Lecture slides (click on preview picture)
<a href="https://ias-uni-siegen.github.io/EMD_course/lecture.pdf" target="_blank" class="image fit"><img src="misc/Lecture_preview.png" alt=""></a>

The covered topics are:
- An initial overview of power electronics
- DC-DC converters


## Exercise tasks (click on preview picture)
<a href="https://ias-uni-siegen.github.io/PE_course/exercise.pdf" target="_blank" class="image fit"><img src="misc/Exercise_preview.png" alt=""></a>

## Exercise tasks including solutions (click on preview picture)
<a href="https://ias-uni-siegen.github.io/PE_course/exercise_with_solution.pdf" target="_blank" class="image fit"><img src="misc/Exercise_with_solution_preview.png" alt=""></a>


## Simulations
- Download KiCad for your operating system at the following link: https://www.kicad.org/download/ and set the system language to English.
- After that create a new project and open the "Schematic Editor"
  
![schematic editor startscreen](https://github.com/user-attachments/assets/d8dc0eb6-3e88-41ce-b3b9-336a76bb9a16)

- Then you have to load the symbol library using the shortcut ‘a’.
  
![symbol library](https://github.com/user-attachments/assets/6b398c3d-81cf-4102-8923-367103336724)

- This symbol library specifically lists the components that are intended for a SPICE circuit simulation. Now you can drag the first components onto the schematic.
- If a circuit is created as shown in the picture below, in which certain sources have time varying properties, these must be set in the internal settings of the components. The same applies to the respective parameters of the components. The internal settings can be opened by double clicking on the symbols. 
  
![buck converter](https://github.com/user-attachments/assets/cfbf9be0-0957-4e0d-b04e-d2d52db9ca50)

- Once the circuit has been created, it can be simulated. To do this, first activate the ‘Inspect’ tab at the top and then the ‘Simulator’ field. The ‘Spice Simulator’ is now opened. The desired analysis must be added there under ‘New Analysis Tab’, such as ‘TRAN Transient Analysis’. Once the desired analysis has been set, the blue arrow must be activated to perform the necessary calculations. In order to be able to view certain curves, these signals must be activated in the right hand field.

![simulation buck converter](https://github.com/user-attachments/assets/2f699cac-79b3-4c94-8332-6910e940948d)


