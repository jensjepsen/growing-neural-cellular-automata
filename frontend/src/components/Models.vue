<script setup lang="ts">
import { reactive } from 'vue'

const state = reactive({
    selected: 1
})
</script>

<template>
    <div>
        <div>
            <select v-model="state.selected">
                <option value="0">Salamander</option>
                <option value="1">Smiley</option>
                <option value="2">Star</option>
                <option value="3">Eye</option>
                <option value="4">Zebrafish</option>
                <option value="5">Butterfly</option>
                <option value="6">Ladybug</option>
                <option value="7">Spiderweb</option>
                <option value="8">Pretzel</option>
                <option value="9">Christmas tree</option>
            </select>
        </div>
        <div class="columns is-justify-content-space-around">
            <Suspense>
                <Model :title="'regenerating'" class="is-flex-grow-0 column" :model-file="state.selected + '_pool=True_damage=True_epochs=500.onnx'" />
            </Suspense>
            <Suspense>
                <Model :title="'stable'" class="is-flex-grow-0 column" :model-file="state.selected + '_pool=True_damage=False_epochs=500.onnx'" />
            </Suspense>
            <Suspense>
                <Model :title="'growing'" class="is-flex-grow-0 column" :model-file="state.selected + '_pool=False_damage=False_epochs=500.onnx'" />
            </Suspense>
        </div>
    </div>
</template>

<style scoped>
    
</style>