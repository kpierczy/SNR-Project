# ================================================================================================================
# @ Author: Krzysztof Pierczyk
# @ Create Time: 2021-01-21 19:19:47
# @ Modified time: 2021-01-21 19:19:51
# @ Description:
#
#       Runs PlotNeuralNet framework on the python file given in the first argument to generate 
#       neural network's visualization. The filename should be given relative to the 
#       $PROJECT_HOME/visualization folder. Example usage:
#
#       viz.bash pipe/pipe.py
#
# @ Note: After running script, the framework will try to open a deleted copy of the .pdf file.
#     It's a known behaviour.
# @ Note: The framework also tries to remove unexisting *.vscodeLog files. It's also a know behaviour.
#     Don't worry about it.
# ================================================================================================================

# Create temporary project's directory
tmp=$PROJECT_HOME/visualization/PlotNeuralNet/tmp
mkdir -p $tmp
dir="$(dirname $1)"
cp $PROJECT_HOME/visualization/$dir/* $tmp/
cd $tmp

# Run the framework
file="$(basename -- $1)"
file="${file%.*}"
bash ../tikzmake.sh $file

# Move result to the initial directory
mv "$file.pdf" $PROJECT_HOME/visualization/$dir/

# Cleanup 
rm -rf $tmp
