import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { DiscoverOcpnComponent } from './discover-ocpn/discover-ocpn.component';
import { LogUploadComponent } from './log-upload/log-upload.component';
import { ObjectModelGeneratorComponent } from './object-model-generator/object-model-generator.component';
import { SimulateProcessComponent } from './simulate-process/simulate-process.component';

const routes: Routes = [
  {path: 'ocel_config', component: LogUploadComponent },
  {path: 'object_model', component: ObjectModelGeneratorComponent },
  {path: 'discover_ocpn', component: DiscoverOcpnComponent},
  {path: 'simulate_process', component: SimulateProcessComponent},

];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
