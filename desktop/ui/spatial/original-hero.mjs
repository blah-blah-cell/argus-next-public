// Original procedural geometry for the public build; no external mesh assets.
export async function createHeroA(T, renderer, {invalidate=()=>{}, onVideoStatus=()=>{}}={}) {
  const shape = new T.Shape();
  shape.moveTo(-1.8,-2); shape.lineTo(-.62,2); shape.lineTo(.62,2);
  shape.lineTo(1.8,-2); shape.lineTo(.98,-2); shape.lineTo(.72,-1.1);
  shape.lineTo(-.72,-1.1); shape.lineTo(-.98,-2); shape.closePath();
  const hole = new T.Path();
  hole.moveTo(-.46,-.35); hole.lineTo(0,1.25); hole.lineTo(.46,-.35); hole.closePath();
  shape.holes.push(hole);
  const geometry = new T.ExtrudeGeometry(shape, {depth:.5, bevelEnabled:true,
    bevelThickness:.1, bevelSize:.1, bevelSegments:3});
  geometry.center();
  const material = new T.MeshPhysicalMaterial({color:0xacc9f6, metalness:.65,
    roughness:.24, clearcoat:1, clearcoatRoughness:.18});
  const group = new T.Group();
  group.add(new T.Mesh(geometry,material));
  group.add(new T.HemisphereLight(0xdcecff,0x273c65,2));
  const light = new T.DirectionalLight(0xffffff,2.5); light.position.set(4,5,7); group.add(light);
  let enabled=false, targetX=0, targetY=0;
  onVideoStatus('unavailable');
  return {
    group,
    resize(){},
    hover(x,y){targetX=y*.12;targetY=x*.18;invalidate();},
    tick(){if(!enabled)return false; group.rotation.x+=(targetX-group.rotation.x)*.12;
      group.rotation.y+=(targetY-group.rotation.y)*.12;
      return Math.abs(targetX-group.rotation.x)+Math.abs(targetY-group.rotation.y)>.001;},
    prepare(){}, leave(){targetX=0;targetY=0;invalidate();},
    setRoughness(value){const n=Number(value);if(Number.isFinite(n)){material.roughness=Math.max(0,Math.min(1,n));invalidate();}},
    setLight(){}, setEnabled(value){enabled=!!value;}, retryVideo(){},
    dispose(){geometry.dispose();material.dispose();}
  };
}
