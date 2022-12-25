import streamlit as st, sys, PIL.Image, torch, numpy as np, imageio.v3, tempfile, os

from model import Model, Cell

checkpoint = sys.argv[1]

m = Model()
m.load_state_dict(torch.load(checkpoint)['state_dict'])

steps = st.slider(min_value=1, max_value=180, value=92, label='Steps')

with torch.no_grad():
    states, _ = m.cell(steps=torch.tensor(180), batch_size=1)
    states = states.numpy()[:, 0, ...]
print('STATES', len(states))

img = states[steps]
with tempfile.TemporaryDirectory() as dir_path:
    path = os.path.join(dir_path, 'test.gif')
    imageio.v3.imwrite(path, states[:steps])
    st.image(path, caption='Evolution')
#PIL.Image.fromarray(img).save('test.png')
#import pdb; pdb.set_trace()
st.image(img, caption=f'Snapshot @ {steps}')
