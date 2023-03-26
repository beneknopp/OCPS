import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';

import { HttpClientModule } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { FontAwesomeModule } from '@fortawesome/angular-fontawesome';
import { NgSelectModule } from '@ng-select/ng-select';
import { NgxGraphModule } from '@swimlane/ngx-graph';
import { NgChartsModule } from 'ng2-charts';
import { DiscoverOcpnComponent } from './discover-ocpn/discover-ocpn.component';
import { LogUploadComponent } from './log-upload/log-upload.component';
import { MessageComponent } from './message/message.component';
import { ObjectModelGeneratorComponent } from './object-model-generator/object-model-generator.component';
import { SimulateProcessComponent } from './simulate-process/simulate-process.component';

@NgModule({
  declarations: [
    AppComponent,
    MessageComponent,
    LogUploadComponent,
    ObjectModelGeneratorComponent,
    SimulateProcessComponent,
    DiscoverOcpnComponent,
  ],
  imports: [
    BrowserModule,
    HttpClientModule,
    NgxGraphModule,
    NgSelectModule,
    FormsModule,
    BrowserAnimationsModule,
    NgChartsModule,
    FontAwesomeModule,
    AppRoutingModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
