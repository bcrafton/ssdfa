
# python imagenet1_viz.py --bias 0 --gpu 0 --alpha 0.001 --dfa 0 --save 1 --name imagenet_bp1 &
# python imagenet1_viz.py --bias 1 --gpu 1 --alpha 0.001 --dfa 0 --save 1 --name imagenet_bp2 &
# python imagenet1_viz.py --bias 0 --gpu 2 --alpha 0.001 --dfa 1 --save 1 --name imagenet_dfa1 &
# python imagenet1_viz.py --bias 1 --gpu 3 --alpha 0.001 --dfa 1 --save 1 --name imagenet_dfa2 &

# python imagenet1_viz.py --bias 0 --gpu 1 --alpha 0.01 --dfa 0 --save 1 --name imagenet_bp2 &
# python imagenet1_viz.py --bias 0 --gpu 3 --alpha 0.01 --dfa 1 --save 1 --name imagenet_dfa2 &

# python imagenet2_viz.py --bias 0 --gpu 1 --alpha 0.001 --dfa 0 --save 1 --name imagenet_bp2 &
# python imagenet2_viz.py --bias 0 --gpu 3 --alpha 0.001 --dfa 1 --save 1 --name imagenet_dfa2 &

# python imagenet3_viz.py --bias 0 --gpu 3 --alpha 0.001 --dfa 0 --save 1 --name imagenet_bp3 &

# python imagenet1_viz_dfa.py --bias 0 --gpu 2 --alpha 0.001 --dfa 1 --save 1 --name imagenet_dfa1_load &

python imagenet1_viz_bp.py --bias 0 --gpu 1 --alpha 0.001 --dfa 1 --save 1 --name imagenet_bp1_load &
