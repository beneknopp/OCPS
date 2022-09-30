import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SimulateProcessComponent } from './simulate-process.component';

describe('SimulateProcessComponent', () => {
  let component: SimulateProcessComponent;
  let fixture: ComponentFixture<SimulateProcessComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ SimulateProcessComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(SimulateProcessComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
