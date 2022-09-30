import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';

import { HttpClientModule } from '@angular/common/http';
import { MessageComponent } from './message/message.component';
import { LogUploadComponent } from './log-upload/log-upload.component';
import { ObjectModelGeneratorComponent } from './object-model-generator/object-model-generator.component';
import { SimulateProcessComponent } from './simulate-process/simulate-process.component';
import { DiscoverOcpnComponent } from './discover-ocpn/discover-ocpn.component';
import { NgxGraphModule } from '@swimlane/ngx-graph';
import { NgSelectModule } from '@ng-select/ng-select';
import { FormsModule } from '@angular/forms';
import { FontAwesomeModule } from '@fortawesome/angular-fontawesome'
import { NgChartsModule } from 'ng2-charts';


@NgModule({
  declarations: [
    AppComponent,
    MessageComponent,
    LogUploadComponent,
    ObjectModelGeneratorComponent,
    SimulateProcessComponent,
    DiscoverOcpnComponent
  ],
  imports: [
    BrowserModule,
    HttpClientModule,
    NgxGraphModule,
    NgSelectModule,
    FormsModule,
    NgChartsModule,
    FontAwesomeModule,
    AppRoutingModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
