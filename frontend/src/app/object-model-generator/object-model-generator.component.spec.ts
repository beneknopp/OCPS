import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ObjectModelGeneratorComponent } from './object-model-generator.component';

describe('ObjectModelGeneratorComponent', () => {
  let component: ObjectModelGeneratorComponent;
  let fixture: ComponentFixture<ObjectModelGeneratorComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ObjectModelGeneratorComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ObjectModelGeneratorComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
